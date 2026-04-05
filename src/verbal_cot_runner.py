import torch
import json
import sys, os
sys.path.insert(0, os.path.abspath("."))
from src.model_utils import load_verbal_cot_model
from src.data_utils import load_split

MAX_LENGTH = 512   # GPT-2 hard context limit

# ---------------------------------------------------------------------------
# Bug in original runner
# ──────────────────────
# Step localization used tokenizer.encode(step_str) in isolation, then tried
# to match the resulting tokens as a subsequence inside the full-sequence
# token list. This fails silently because GPT-2 uses byte-pair encoding (BPE):
# the tokenization of a substring depends on its surrounding context.
# A word at the start of an isolated encode gets a different token ID than
# the same word mid-sequence (GPT-2 uses "Ġ" prefix tokens for words after
# a space). The result is `find_subsequence` returning -1 for many steps,
# producing None reps and a flood of WARNING messages.
#
# Fix
# ───
# Tokenize full_text once with return_offsets_mapping=True. Each token gets
# a (char_start, char_end) span into the original string. To locate step k,
# find its character span in full_text, then pick the last token whose
# char_end falls within that span. This is exact and context-aware.
#
# Note: return_offsets_mapping is not natively supported by GPT-2's fast
# tokenizer in all versions. If it raises, we fall back to the original
# approach with one targeted fix: prepend a space to match in-context
# BPE tokenization (GPT-2 encodes interior words with a leading Ġ).
# ---------------------------------------------------------------------------

# ── Layer index to hook ─────────────────────────────────────────────────────
# GPT-2 medium has 24 layers (h[0]..h[23]).
# We hook the final layer (h[-1] = h[23]) to get the last residual stream
# activation, matching what coconut_runner.py does for a controlled comparison.
LAYER = -1


def extract_verbal_representations(
    data_path="data/prontoqa_split.csv",
    split="test",
    output_path="data/verbal_reps.json",
    device="cuda",
    layer=LAYER,
):
    model, tokenizer = load_verbal_cot_model(device=device)
    model.eval()

    df = load_split(data_path, split=split)

    results        = []
    n_none_total   = 0

    for _, row in df.iterrows():
        steps     = row["steps"]                      # list[str]
        steps_text = "\n".join(steps)
        full_text  = row["question"] + "\n" + steps_text

        # ── Tokenize with offset mapping ────────────────────────────────
        # return_offsets_mapping gives each token its (char_start, char_end)
        # span into full_text, letting us locate steps by character position
        # rather than re-tokenizing substrings out of context.
        try:
            inputs = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
                return_offsets_mapping=True,
            )
            offset_mapping = inputs.pop("offset_mapping")[0].tolist()   # (seq_len, 2)
            use_offsets    = True
        except Exception:
            # Fallback: fast tokenizer not available or offset_mapping unsupported
            inputs         = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
            )
            offset_mapping = None
            use_offsets    = False

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # ── Register hook on the final transformer layer ─────────────────
        captured = {}
        def hook_fn(module, input, output):
            captured["hidden"] = output[0].detach().cpu()   # (batch, seq, d_model)

        handle = model.transformer.h[layer].register_forward_hook(hook_fn)
        with torch.no_grad():
            model(**inputs)
        handle.remove()

        hidden = captured["hidden"][0]   # (seq_len, d_model)

        # ── Locate each step and extract its last-token hidden state ─────
        step_reps    = []
        search_start = 0   # character offset (offset mode) or token index (fallback)

        for k, step_str in enumerate(steps):
            if use_offsets:
                tok_pos = _locate_step_by_offset(
                    step_str, full_text, offset_mapping,
                    search_start_char=search_start,
                )
                if tok_pos is None:
                    step_reps.append(None)
                    n_none_total += 1
                    print(
                        f"WARNING: could not locate step {k+1} in problem {row.name} "
                        f"(offset mode). Step text: {step_str!r:.60}",
                        flush=True,
                    )
                    continue
                step_reps.append(hidden[tok_pos].tolist())
                # Advance character search past this step's content
                char_start = full_text.find(step_str, search_start)
                search_start = char_start + len(step_str)

            else:
                # Fallback: token subsequence matching with BPE-aware encoding.
                # FIX vs original: prepend " " (space) when k > 0 so GPT-2
                # produces the Ġ-prefixed tokens that appear in-context.
                tok_pos, advance = _locate_step_by_tokens(
                    step_str, k,
                    tokenizer,
                    tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu().tolist()),
                    search_start_tok=search_start,
                )
                if tok_pos is None:
                    step_reps.append(None)
                    n_none_total += 1
                    print(
                        f"WARNING: could not locate step {k+1} in problem {row.name} "
                        f"(token mode). Step text: {step_str!r:.60}",
                        flush=True,
                    )
                    continue
                step_reps.append(hidden[tok_pos].tolist())
                search_start = advance

        results.append({
            "problem_id":         int(row.name),
            "n_hops":             int(row["n_hops"]),
            "split":              row["split"],
            "ground_truth_steps": steps,
            "final_answer":       row["answer"],
            "step_reps":          step_reps,   # list[list[float] | None], len == len(steps)
        })

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} problems → {output_path}")
    print(f"Total None step_reps: {n_none_total}")
    if n_none_total > 0:
        print(
            "  ↳ Non-zero Nones usually mean the step text was truncated by "
            "MAX_LENGTH or the tokenizer doesn't support offset_mapping. "
            "Try increasing MAX_LENGTH or upgrading transformers."
        )
    return results


# ── Helpers ──────────────────────────────────────────────────────────────────

def _locate_step_by_offset(step_str, full_text, offset_mapping, search_start_char=0):
    """
    Find the last token whose character span falls within `step_str` in `full_text`.

    Returns the token index (into offset_mapping / hidden states), or None if
    the step was truncated or not found.
    """
    char_start = full_text.find(step_str, search_start_char)
    if char_start == -1:
        return None
    char_end = char_start + len(step_str)

    # Walk backwards from the end of the sequence to find the last token
    # whose end offset is <= char_end and whose start >= char_start.
    best_idx = None
    for i, (s, e) in enumerate(offset_mapping):
        if s >= char_start and e <= char_end and e > s:
            best_idx = i   # keep updating; we want the last such token
    return best_idx


def _locate_step_by_tokens(step_str, step_idx, tokenizer, tokens, search_start_tok=0):
    """
    Fallback token-matching that accounts for GPT-2 BPE context dependence.

    GPT-2 adds a Ġ prefix to words that follow a space. When `step_idx > 0`
    the step appears mid-sequence preceded by a newline, so we prepend " "
    (space) before encoding to get the Ġ-prefixed tokens that appear in context.

    Returns (last_content_token_idx, next_search_start) or (None, search_start_tok).
    """
    # Encode with the space prefix to match in-context BPE tokenization
    prefix      = " " if step_idx > 0 else ""
    encoded_nl  = tokenizer.convert_ids_to_tokens(
        tokenizer.encode(prefix + step_str + "\n", add_special_tokens=False)
    )
    encoded_nonl = tokenizer.convert_ids_to_tokens(
        tokenizer.encode(prefix + step_str, add_special_tokens=False)
    )

    pos = _find_subsequence(tokens, encoded_nl, start=search_start_tok)
    if pos != -1:
        # Last content token is second-to-last (newline is the final token)
        last_content = pos + len(encoded_nl) - 2
        return last_content, pos + 1

    pos = _find_subsequence(tokens, encoded_nonl, start=search_start_tok)
    if pos != -1:
        last_content = pos + len(encoded_nonl) - 1
        return last_content, pos + 1

    # Also try without the space prefix in case step appears at sequence start
    for enc in [
        tokenizer.convert_ids_to_tokens(tokenizer.encode(step_str + "\n", add_special_tokens=False)),
        tokenizer.convert_ids_to_tokens(tokenizer.encode(step_str, add_special_tokens=False)),
    ]:
        pos = _find_subsequence(tokens, enc, start=search_start_tok)
        if pos != -1:
            last_content = pos + len(enc) - (2 if enc[-1] == "Ċ" else 1)  # Ċ = \n in GPT-2
            return last_content, pos + 1

    return None, search_start_tok


def _find_subsequence(token_list, subseq, start=0):
    """Return start index of first occurrence of subseq at or after `start`, or -1."""
    n, m = len(token_list), len(subseq)
    for i in range(start, n - m + 1):
        if token_list[i:i+m] == subseq:
            return i
    return -1


if __name__ == "__main__":
    extract_verbal_representations()