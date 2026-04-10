import torch
import json
import sys, os
sys.path.insert(0, os.path.abspath("."))
from src.model_utils import load_verbal_cot_model
from src.data_utils import load_split

MAX_LENGTH = 512

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
        steps     = row["steps"]
        steps_text = "\n".join(steps)
        full_text  = row["question"] + "\n" + steps_text

        try:
            inputs = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
                return_offsets_mapping=True,
            )
            offset_mapping = inputs.pop("offset_mapping")[0].tolist()
            use_offsets    = True
        except Exception:
            inputs         = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
            )
            offset_mapping = None
            use_offsets    = False

        inputs = {k: v.to(device) for k, v in inputs.items()}

        captured = {}
        def hook_fn(module, input, output):
            captured["hidden"] = output[0].detach().cpu()

        handle = model.transformer.h[layer].register_forward_hook(hook_fn)
        with torch.no_grad():
            model(**inputs)
        handle.remove()

        hidden = captured["hidden"][0]

        step_reps    = []
        search_start = 0

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
                char_start = full_text.find(step_str, search_start)
                search_start = char_start + len(step_str)

            else:
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
            "step_reps":          step_reps,
        })

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} problems -> {output_path}")
    print(f"Total None step_reps: {n_none_total}")
    if n_none_total > 0:
        print(
            "  -> Non-zero Nones usually mean the step text was truncated by "
            "MAX_LENGTH or the tokenizer doesn't support offset_mapping. "
            "Try increasing MAX_LENGTH or upgrading transformers."
        )
    return results

def _locate_step_by_offset(step_str, full_text, offset_mapping, search_start_char=0):
    char_start = full_text.find(step_str, search_start_char)
    if char_start == -1:
        return None
    char_end = char_start + len(step_str)

    best_idx = None
    for i, (s, e) in enumerate(offset_mapping):
        if s >= char_start and e <= char_end and e > s:
            best_idx = i
    return best_idx

def _locate_step_by_tokens(step_str, step_idx, tokenizer, tokens, search_start_tok=0):
    prefix      = " " if step_idx > 0 else ""
    encoded_nl  = tokenizer.convert_ids_to_tokens(
        tokenizer.encode(prefix + step_str + "\n", add_special_tokens=False)
    )
    encoded_nonl = tokenizer.convert_ids_to_tokens(
        tokenizer.encode(prefix + step_str, add_special_tokens=False)
    )

    pos = _find_subsequence(tokens, encoded_nl, start=search_start_tok)
    if pos != -1:
        last_content = pos + len(encoded_nl) - 2
        return last_content, pos + 1

    pos = _find_subsequence(tokens, encoded_nonl, start=search_start_tok)
    if pos != -1:
        last_content = pos + len(encoded_nonl) - 1
        return last_content, pos + 1

    for enc in [
        tokenizer.convert_ids_to_tokens(tokenizer.encode(step_str + "\n", add_special_tokens=False)),
        tokenizer.convert_ids_to_tokens(tokenizer.encode(step_str, add_special_tokens=False)),
    ]:
        pos = _find_subsequence(tokens, enc, start=search_start_tok)
        if pos != -1:
            last_content = pos + len(enc) - (2 if enc[-1] == "Ċ" else 1)
            return last_content, pos + 1

    return None, search_start_tok

def _find_subsequence(token_list, subseq, start=0):
    n, m = len(token_list), len(subseq)
    for i in range(start, n - m + 1):
        if token_list[i:i+m] == subseq:
            return i
    return -1

if __name__ == "__main__":
    extract_verbal_representations()