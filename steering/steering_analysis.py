"""
steering_analysis.py

Three analyses:
  1. Logistic regression — is correct/wrong linearly separable at each stage?
  2. Word2Vec-style parallelogram plot:
       X-axis = projection onto steering vector (correct/wrong direction)
       Y-axis = first PC of the residual (semantic content / topic)
     Mirrors the king-man+woman=queen analogy: if the vector generalises,
     pairs should form consistent horizontal offsets regardless of their
     vertical (semantic) position.
  3. Norm diagnosis summary table.

Run from ~/coconut-cot:
  python steering_analysis.py --config steering/steering_config.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coconut"))

import torch
import yaml
from transformers import GPT2LMHeadModel, GPT2Tokenizer

LATENT_TOKEN_ID = 50257
START_LATENT_ID = 50258
END_LATENT_ID   = 50259
EOS_TOKEN_ID    = 50256
VOCAB_SIZE      = 50260

DARK  = "#0d1117"
SURF  = "#161b22"
GRID  = "#21262d"
TEXT  = "#e6edf3"
MUTED = "#8b949e"
GREEN = "#3fb950"
RED   = "#f85149"
AMBER = "#d29922"
PURPL = "#bc8cff"


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_tokenizer():
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.add_special_tokens({"additional_special_tokens":
        ["<|latent|>", "<|start-latent|>", "<|end-latent|>"]})
    tok.pad_token = tok.eos_token
    return tok


def load_model(cfg, device):
    from coconut import Coconut
    base = GPT2LMHeadModel.from_pretrained("gpt2")
    base.resize_token_embeddings(VOCAB_SIZE)
    model = Coconut(
        base_causallm=base,
        latent_token_id=LATENT_TOKEN_ID,
        start_latent_id=START_LATENT_ID,
        end_latent_id=END_LATENT_ID,
        eos_token_id=EOS_TOKEN_ID,
    )
    ckpt = torch.load(os.path.expanduser(cfg["model_path"]), map_location="cpu")
    if "base_causallm" in ckpt:
        ckpt = ckpt["base_causallm"]
    ckpt = {k.replace("base_causallm.", ""): v for k, v in ckpt.items()}
    model.base_causallm.load_state_dict(ckpt, strict=False)
    model.to(device).eval()
    return model


def build_input(tokenizer, question, n_latent, device):
    ids = torch.tensor(
        [tokenizer.encode(question) + [START_LATENT_ID] +
         [LATENT_TOKEN_ID] * n_latent + [END_LATENT_ID]],
        dtype=torch.long, device=device,
    )
    return (ids, torch.ones_like(ids), ids.clone(),
            torch.arange(0, ids.shape[1], dtype=torch.long,
                         device=device).unsqueeze(0))


def get_thought(model, tokenizer, question, n_latent, device, capture_pass):
    input_ids, attn, labels, pos = build_input(tokenizer, question, n_latent, device)
    captured = [None]
    li = (input_ids == model.latent_token_id).nonzero()
    ll = [[idx[1].item() for idx in li if idx[0] == i]
          for i in range(input_ids.shape[0])]
    mn  = max(len(l) for l in ll)
    emb = model.embedding(input_ids)
    ncr = (0, li[:, 1].min().item())
    kv  = None

    for pi in range(mn):
        if kv is None:
            out = model.base_causallm(
                inputs_embeds=emb[:, ncr[0]:ncr[1], :],
                attention_mask=attn[:, ncr[0]:ncr[1]],
                position_ids=pos[:, ncr[0]:ncr[1]],
                output_hidden_states=True)
            offset = 0
        else:
            pkv = [(k[:, :, :ncr[0], :], v[:, :, :ncr[0], :]) for k, v in kv]
            out = model.base_causallm(
                inputs_embeds=emb[:, ncr[0]:ncr[1], :],
                attention_mask=attn[:, :ncr[1]],
                position_ids=pos[:, ncr[0]:ncr[1]],
                past_key_values=pkv,
                output_hidden_states=True)
            offset = ncr[0]

        ncr = (ncr[1], input_ids.shape[1] if pi + 1 >= mn else ncr[1] + 1)
        hs  = out.hidden_states[-1]
        kv  = out.past_key_values
        tl  = [[emb[b, p, :] for p in range(emb.shape[1])]
               for b in range(emb.shape[0])]

        for b, ml in enumerate(ll):
            if len(ml) > pi:
                tok_idx = ml[pi]
                th = hs[b, tok_idx - 1 - offset, :]
                if pi == capture_pass:
                    captured[0] = th.detach().cpu()
                tl[b][tok_idx] = th

        emb = torch.stack([torch.stack(tl[b]) for b in range(emb.shape[0])])

    return captured[0]


def neg_question(sample):
    q    = sample["question"]
    syms = sample.get("idx_to_symbol", [])
    t    = sample.get("target")
    n    = sample.get("neg_target")
    tw   = syms[t] if t is not None and t < len(syms) else ""
    nw   = syms[n] if n is not None and n < len(syms) else ""
    if tw and nw and f"{tw} or {nw}" in q:
        q = q.replace(f"{tw} or {nw}", f"{nw} or {tw}")
    return q


def collect(model, tokenizer, data, n_latent, device, capture_pass, n, tag):
    correct, wrong = [], []
    for i, s in enumerate(data[:n]):
        with torch.no_grad():
            hc = get_thought(model, tokenizer, s["question"],
                             n_latent, device, capture_pass)
            hw = get_thought(model, tokenizer, neg_question(s),
                             n_latent, device, capture_pass)
        if hc is not None:
            correct.append(hc)
        if hw is not None:
            wrong.append(hw)
        if (i + 1) % 50 == 0:
            print(f"    {tag}: {i+1}/{min(n, len(data))}")
    return correct, wrong


def plot_analogy(pos, vec, tc_tr, tw_tr, tc_te, tw_te, out_dir):
    """
    Word2Vec-style analogy parallelogram plot.

    X = projection onto the steering (correct-minus-wrong) vector
    Y = first principal component of the residual (captures semantic variation)

    In word2vec space king - man + woman = queen because the gender
    direction is consistent across semantically different pairs.
    Here: if the vector generalises, every (wrong, correct) pair should
    have the same horizontal offset regardless of its vertical position.
    """
    all_v = tc_tr + tw_tr + tc_te + tw_te
    mat   = torch.stack(all_v).float().numpy()

    # Steering direction (unit vector)
    sv   = vec.float().numpy()
    sv_u = sv / (np.linalg.norm(sv) + 1e-8)

    # X: project onto steering vector
    x_all = mat @ sv_u

    # Y: PC1 of residual (semantic axis)
    residual = mat - np.outer(x_all, sv_u)
    pca      = PCA(n_components=1)
    y_all    = pca.fit_transform(residual).squeeze()

    n_tr = len(tc_tr)
    n_te = len(tc_te)

    xc_tr = x_all[:n_tr];           yc_tr = y_all[:n_tr]
    xw_tr = x_all[n_tr:2*n_tr];     yw_tr = y_all[n_tr:2*n_tr]
    xc_te = x_all[2*n_tr:2*n_tr+n_te]; yc_te = y_all[2*n_tr:2*n_tr+n_te]
    xw_te = x_all[2*n_tr+n_te:];    yw_te = y_all[2*n_tr+n_te:]

    diffs     = xc_te - xw_te
    mean_dir  = float(np.mean(diffs))
    frac_pos  = float(np.mean(diffs > 0))
    std_dir   = float(np.std(diffs))

    fig, ax = plt.subplots(figsize=(11, 7.5))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)
    ax.grid(color=GRID, linewidth=0.5, zorder=0)

    # ── train cloud (faint) ──────────────────────────────────────────────
    for i in range(min(100, n_tr)):
        ax.plot([xw_tr[i], xc_tr[i]], [yw_tr[i], yc_tr[i]],
                color=MUTED, alpha=0.08, linewidth=0.6, zorder=1)
    ax.scatter(xc_tr, yc_tr, color=GREEN, alpha=0.2, s=14, zorder=2)
    ax.scatter(xw_tr, yw_tr, color=RED,   alpha=0.2, s=14, zorder=2)

    # ── test parallelograms (arrows) ─────────────────────────────────────
    for i in range(min(50, n_te)):
        ax.annotate("", xy=(xc_te[i], yc_te[i]),
                    xytext=(xw_te[i], yw_te[i]),
                    arrowprops=dict(arrowstyle="-|>",
                                   color=AMBER, lw=0.9, alpha=0.45),
                    zorder=3)
    ax.scatter(xc_te, yc_te, color=GREEN, s=55, marker="^",
               edgecolors="white", linewidths=0.5, zorder=5,
               label="test correct")
    ax.scatter(xw_te, yw_te, color=RED,   s=55, marker="v",
               edgecolors="white", linewidths=0.5, zorder=5,
               label="test wrong")

    # ── mean parallelogram (bold gold arrow) ─────────────────────────────
    mx_c = float(np.mean(xc_te)); my_c = float(np.mean(yc_te))
    mx_w = float(np.mean(xw_te)); my_w = float(np.mean(yw_te))
    ax.annotate("", xy=(mx_c, my_c), xytext=(mx_w, my_w),
                arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=2.8),
                zorder=6)
    ax.scatter([mx_c], [my_c], color=AMBER, s=150, zorder=7, marker="*")

    # ── steering vector direction bar ────────────────────────────────────
    xlim = ax.get_xlim()
    bar_y = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.06
    bar_len = (xlim[1] - xlim[0]) * 0.18
    bar_x   = xlim[0] + (xlim[1] - xlim[0]) * 0.72
    ax.annotate("", xy=(bar_x + bar_len, bar_y),
                xytext=(bar_x, bar_y),
                arrowprops=dict(arrowstyle="-|>", color=PURPL, lw=2.2))
    ax.text(bar_x + bar_len / 2, bar_y + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.025,
            "steering vector", color=PURPL, fontsize=8,
            ha="center", va="bottom")

    # ── labels ───────────────────────────────────────────────────────────
    ax.set_xlabel(
        "← wrong                   correct →\n"
        "(dot product with steering vector)",
        color=MUTED, fontsize=10, labelpad=8)
    ax.set_ylabel(
        "Semantic axis\n(PC1 of residual — question topic / context)",
        color=MUTED, fontsize=10, labelpad=8)
    ax.set_title(
        f"L{pos} thought space  —  Word2Vec-style analogy structure\n"
        f"Mean offset: {mean_dir:+.4f}  ±{std_dir:.4f}   "
        f"fraction correct > wrong: {frac_pos:.0%}",
        color=TEXT, fontsize=11, pad=12)
    ax.tick_params(colors=MUTED, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color(GRID)

    legend = ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="^", color="w",
                       markerfacecolor=GREEN, markersize=8, label="test correct"),
            plt.Line2D([0], [0], marker="v", color="w",
                       markerfacecolor=RED, markersize=8, label="test wrong"),
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=GREEN, alpha=0.4, markersize=6,
                       label="train correct (faint)"),
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=RED, alpha=0.4, markersize=6,
                       label="train wrong (faint)"),
            plt.Line2D([0], [0], color=AMBER, lw=1.5,
                       label="test pair offset (arrow)"),
        ],
        loc="upper left", fontsize=8,
        facecolor=SURF, edgecolor=GRID, labelcolor=TEXT,
    )

    # ── annotation box ───────────────────────────────────────────────────
    note = (
        "Ideal (word2vec-like):\n"
        "  all arrows horizontal, same length\n"
        "  pairs stack vertically by topic\n"
        "  gold star sits right of red cluster"
    )
    ax.text(0.985, 0.97, note, transform=ax.transAxes,
            color=MUTED, fontsize=7.5, ha="right", va="top",
            bbox=dict(facecolor=SURF, edgecolor=GRID,
                      boxstyle="round,pad=0.45", alpha=0.9))

    plt.tight_layout()
    path = out_dir / f"analogy_L{pos}.png"
    plt.savefig(path, dpi=150, facecolor=DARK)
    plt.close()
    print(f"  Saved {path}")
    return mean_dir, frac_pos, std_dir


def run(cfg):
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model     = load_model(cfg, device)
    tokenizer = load_tokenizer()
    n_latent  = 6

    vec_path = Path(os.path.expanduser(cfg["output_dir"])) / "steering_vectors.pt"
    vectors  = torch.load(vec_path, map_location="cpu")

    with open(os.path.expanduser(cfg["train_path"])) as f:
        all_data = json.load(f)
    max_pos  = max(cfg["latent_positions"])
    all_data = [d for d in all_data if len(d.get("steps", [])) >= max_pos]

    n_train  = cfg["n_contrast"]
    train    = all_data[:n_train]
    test     = all_data[n_train: n_train + 250]

    out_dir  = Path(os.path.expanduser(cfg["output_dir"])) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []

    for pos in cfg["latent_positions"]:
        cp  = pos - 1
        vec = vectors[pos]
        print(f"\n=== L{pos} ===  vec_norm={vec.norm():.4f}")

        print("  Collecting train representations...")
        tc_tr, tw_tr = collect(model, tokenizer, train, n_latent,
                                device, cp, n=150, tag="train")

        print("  Collecting test representations...")
        tc_te, tw_te = collect(model, tokenizer, test,  n_latent,
                                device, cp, n=80, tag="test")

        if not tc_tr or not tw_tr or not tc_te or not tw_te:
            print("  Skipping — not enough reps"); continue

        t_norms = [h.norm().item() for h in tc_tr[:30]]
        mean_tn  = float(np.mean(t_norms))
        ratio    = vec.norm().item() / mean_tn
        needed   = mean_tn / vec.norm().item()

        # Logistic regression separability
        X   = torch.stack(tc_tr + tw_tr).float().numpy()
        y   = np.array([1] * len(tc_tr) + [0] * len(tw_tr))
        clf = LogisticRegression(max_iter=500, C=1.0).fit(X, y)
        Xte = torch.stack(tc_te + tw_te).float().numpy()
        yte = np.array([1] * len(tc_te) + [0] * len(tw_te))
        clf_acc = clf.score(Xte, yte)

        print(f"  Thought norm:     {mean_tn:.2f}")
        print(f"  Vec/thought:      {ratio:.5f}")
        print(f"  Alpha for 1:1:    {needed:.1f}")
        print(f"  LogReg test acc:  {clf_acc:.3f}")

        mean_dir, frac_pos, std_dir = plot_analogy(
            pos, vec, tc_tr, tw_tr, tc_te, tw_te, out_dir)

        summary.append(dict(
            pos=pos, vec_norm=vec.norm().item(), thought_norm=mean_tn,
            ratio=ratio, needed_alpha=needed, clf_acc=clf_acc,
            mean_dir=mean_dir, frac_pos=frac_pos, std_dir=std_dir,
        ))

    # ── summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"{'L':>3}  {'vec_norm':>9}  {'thought':>8}  {'ratio':>8}  "
          f"{'alpha_1:1':>9}  {'clf_acc':>7}  {'mean_Δx':>8}  "
          f"{'frac>0':>7}")
    print("-" * 80)
    for r in summary:
        print(f"  {r['pos']}  {r['vec_norm']:9.4f}  {r['thought_norm']:8.2f}  "
              f"{r['ratio']:8.5f}  {r['needed_alpha']:9.1f}  "
              f"{r['clf_acc']:7.3f}  {r['mean_dir']:8.4f}  {r['frac_pos']:7.0%}")

    print("\nDiagnosis:")
    for r in summary:
        print(f"\n  L{r['pos']}:")
        if r["clf_acc"] > 0.65:
            print("    ✓ correct/wrong IS linearly separable "
                  f"(acc={r['clf_acc']:.0%}) — space has structure")
        else:
            print(f"    ✗ correct/wrong NOT separable (acc={r['clf_acc']:.0%}) "
                  "— contrastive signal too weak")
        if r["frac_pos"] > 0.55:
            print("    ✓ vector direction aligns with correct→wrong offset")
        else:
            print(f"    ✗ vector direction is random/wrong "
                  f"({r['frac_pos']:.0%} of test pairs point right)")
        print(f"    ⚠ effective steering requires alpha ≈ {r['needed_alpha']:.0f} "
              f"(current sweep tops at 40)")

    print(f"\nFigures: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(load_config(args.config))