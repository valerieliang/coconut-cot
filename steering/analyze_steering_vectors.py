"""
analyze_steering_vectors.py

Three-part analysis:
  1. Directionality check — does the steering vector point from wrong→correct
     answers? Measured by cosine sim between (correct - wrong) pairs on a
     held-out test set.
  2. PCA plot — project thought representations into 2D, plot correct/wrong
     pairs and the steering vector projection, train vs test split coloured
     separately.
  3. Norm diagnostics — compare thought vector norms to steering vector norms
     to understand why high alpha is needed.

Run from ~/coconut-cot:
  python analyze_steering_vectors.py --config steering/steering_config.yaml
"""

import argparse, json, os, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent / "coconut"))

import torch
import yaml
from transformers import GPT2LMHeadModel, GPT2Tokenizer

LATENT_TOKEN_ID = 50257
START_LATENT_ID = 50258
END_LATENT_ID   = 50259
EOS_TOKEN_ID    = 50256
VOCAB_SIZE      = 50260


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
    model = Coconut(base_causallm=base,
        latent_token_id=LATENT_TOKEN_ID, start_latent_id=START_LATENT_ID,
        end_latent_id=END_LATENT_ID, eos_token_id=EOS_TOKEN_ID)
    ckpt = torch.load(os.path.expanduser(cfg["model_path"]), map_location="cpu")
    if "base_causallm" in ckpt: ckpt = ckpt["base_causallm"]
    ckpt = {k.replace("base_causallm.", ""): v for k, v in ckpt.items()}
    model.base_causallm.load_state_dict(ckpt, strict=False)
    model.to(device).eval()
    return model


def build_input(tokenizer, question, n_latent, device):
    q_ids = tokenizer.encode(question)
    seq = q_ids + [START_LATENT_ID] + [LATENT_TOKEN_ID]*n_latent + [END_LATENT_ID]
    ids = torch.tensor([seq], dtype=torch.long, device=device)
    return (ids, torch.ones_like(ids), ids.clone(),
            torch.arange(0, ids.shape[1], dtype=torch.long,
                         device=device).unsqueeze(0))


def get_thought(model, tokenizer, question, n_latent, device, capture_pass):
    """Extract the continuous thought vector at capture_pass (0-indexed)."""
    input_ids, attn, labels, pos = build_input(tokenizer, question, n_latent, device)
    captured = [None]

    latent_idx = (input_ids == model.latent_token_id).nonzero()
    latent_lists = [[idx[1].item() for idx in latent_idx if idx[0]==i]
                    for i in range(input_ids.shape[0])]
    max_n = max(len(l) for l in latent_lists)
    inputs_embeds = model.embedding(input_ids)
    ncr = (0, latent_idx[:,1].min().item())
    kv = None

    for pi in range(max_n):
        if kv is None:
            out = model.base_causallm(
                inputs_embeds=inputs_embeds[:,ncr[0]:ncr[1],:],
                attention_mask=attn[:,ncr[0]:ncr[1]],
                position_ids=pos[:,ncr[0]:ncr[1]],
                output_hidden_states=True)
            offset = 0
        else:
            pkv = [(k[:,:,:ncr[0],:],v[:,:,:ncr[0],:]) for k,v in kv]
            out = model.base_causallm(
                inputs_embeds=inputs_embeds[:,ncr[0]:ncr[1],:],
                attention_mask=attn[:,:ncr[1]],
                position_ids=pos[:,ncr[0]:ncr[1]],
                past_key_values=pkv, output_hidden_states=True)
            offset = ncr[0]

        ncr = (ncr[1], input_ids.shape[1] if pi+1>=max_n else ncr[1]+1)
        hs = out.hidden_states[-1]
        kv = out.past_key_values

        tl = [[inputs_embeds[b,p,:] for p in range(inputs_embeds.shape[1])]
              for b in range(inputs_embeds.shape[0])]
        for b, ml in enumerate(latent_lists):
            if len(ml) > pi:
                tok_idx = ml[pi]
                thought = hs[b, tok_idx-1-offset, :]
                if pi == capture_pass:
                    captured[0] = thought.detach().cpu()
                tl[b][tok_idx] = thought
        inputs_embeds = torch.stack([torch.stack(tl[b])
                                     for b in range(inputs_embeds.shape[0])])

    return captured[0]


def build_neg_question(sample):
    q = sample["question"]
    syms = sample.get("idx_to_symbol", [])
    t = sample.get("target"); n = sample.get("neg_target")
    tw = syms[t] if t is not None and t < len(syms) else ""
    nw = syms[n] if n is not None and n < len(syms) else ""
    if tw and nw and f"{tw} or {nw}" in q:
        q = q.replace(f"{tw} or {nw}", f"{nw} or {tw}")
    return q


def cosine_sim(a, b):
    a, b = a.float(), b.float()
    return (a @ b / (a.norm() * b.norm() + 1e-8)).item()


def analyze(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model     = load_model(cfg, device)
    tokenizer = load_tokenizer()
    n_latent  = 6

    vectors = torch.load(os.path.expanduser(cfg["output_dir"]) +
                         "/steering_vectors.pt", map_location="cpu")

    with open(os.path.expanduser(cfg["train_path"])) as f:
        all_data = json.load(f)

    max_pos = max(cfg["latent_positions"])
    all_data = [d for d in all_data if len(d.get("steps",[])) >= max_pos]

    n_train = cfg["n_contrast"]
    train_data = all_data[:n_train]
    test_data  = all_data[n_train: n_train + 200]

    out_dir = Path(os.path.expanduser(cfg["output_dir"])) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    for pos in cfg["latent_positions"]:
        capture_pass = pos - 1
        vec = vectors[pos]

        print(f"\n=== Position L{pos} ===")
        print(f"  Steering vector norm:  {vec.norm():.4f}")

        # ── 1. Directionality on test set ──────────────────────────────────
        print(f"  Collecting test representations ({len(test_data)} samples)...")
        correct_test, wrong_test = [], []
        for s in test_data[:100]:
            with torch.no_grad():
                hc = get_thought(model, tokenizer, s["question"],
                                 n_latent, device, capture_pass)
                hw = get_thought(model, tokenizer, build_neg_question(s),
                                 n_latent, device, capture_pass)
            if hc is not None: correct_test.append(hc)
            if hw is not None: wrong_test.append(hw)

        diffs_test = [c - w for c, w in zip(correct_test, wrong_test)]
        sims = [cosine_sim(d, vec) for d in diffs_test]
        mean_sim = np.mean(sims)
        frac_pos  = np.mean([s > 0 for s in sims])
        print(f"  Directionality (cosine sim to steering vec):")
        print(f"    mean={mean_sim:.4f}  frac_positive={frac_pos:.2f}")
        print(f"    (1.0 = perfect alignment, >0.5 = good direction)")

        thought_norms = [h.norm().item() for h in correct_test[:20]]
        print(f"  Thought vector norms (first 20): mean={np.mean(thought_norms):.2f}")
        print(f"  Steering/thought norm ratio: {vec.norm().item()/np.mean(thought_norms):.4f}")
        print(f"  → Alpha needed for 1:1 perturbation: {np.mean(thought_norms)/vec.norm().item():.1f}")

        # ── 2. Train thought vectors for PCA ──────────────────────────────
        print(f"  Collecting train representations...")
        correct_train, wrong_train = [], []
        for s in train_data[:150]:
            with torch.no_grad():
                hc = get_thought(model, tokenizer, s["question"],
                                 n_latent, device, capture_pass)
                hw = get_thought(model, tokenizer, build_neg_question(s),
                                 n_latent, device, capture_pass)
            if hc is not None: correct_train.append(hc)
            if hw is not None: wrong_train.append(hw)

        # ── 3. PCA projection ──────────────────────────────────────────────
        all_vecs = (correct_train + wrong_train +
                    correct_test[:50] + wrong_test[:50] + [vec])
        mat = torch.stack(all_vecs).float().numpy()

        pca = PCA(n_components=2)
        proj = pca.fit_transform(mat)
        var  = pca.explained_variance_ratio_

        n_tr = len(correct_train)
        p_correct_train = proj[:n_tr]
        p_wrong_train   = proj[n_tr:2*n_tr]
        p_correct_test  = proj[2*n_tr:2*n_tr+50]
        p_wrong_test    = proj[2*n_tr+50:2*n_tr+100]
        p_vec           = proj[-1]

        fig, ax = plt.subplots(figsize=(9, 7))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")

        # train pairs
        ax.scatter(p_correct_train[:,0], p_correct_train[:,1],
                   c="#4ade80", alpha=0.5, s=18, label="train correct", zorder=3)
        ax.scatter(p_wrong_train[:,0], p_wrong_train[:,1],
                   c="#f87171", alpha=0.5, s=18, label="train wrong", zorder=3)

        # test pairs
        ax.scatter(p_correct_test[:,0], p_correct_test[:,1],
                   c="#86efac", alpha=0.85, s=40, marker="^",
                   label="test correct", zorder=4)
        ax.scatter(p_wrong_test[:,0], p_wrong_test[:,1],
                   c="#fca5a5", alpha=0.85, s=40, marker="^",
                   label="test wrong", zorder=4)

        # pair lines for first 30 test pairs
        for i in range(min(30, len(p_correct_test))):
            ax.plot([p_correct_test[i,0], p_wrong_test[i,0]],
                    [p_correct_test[i,1], p_wrong_test[i,1]],
                    c="#94a3b8", alpha=0.25, lw=0.7, zorder=2)

        # steering vector as arrow from origin-ish
        cx = proj[:-1,0].mean(); cy = proj[:-1,1].mean()
        scale = (proj[:-1,0].std() + proj[:-1,1].std()) * 0.6
        vdir  = p_vec - proj[:-1].mean(axis=0)
        vnorm = np.linalg.norm(vdir) + 1e-8
        ax.annotate("", xy=(cx + vdir[0]/vnorm*scale, cy + vdir[1]/vnorm*scale),
                    xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="->", color="#facc15", lw=2.5))
        ax.text(cx + vdir[0]/vnorm*scale*1.08, cy + vdir[1]/vnorm*scale*1.08,
                f"steering vec\n(cos_sim={mean_sim:.2f})",
                color="#facc15", fontsize=8, ha="center")

        ax.set_title(f"Latent position L{pos} — thought space PCA\n"
                     f"PC1={var[0]:.1%}  PC2={var[1]:.1%}  "
                     f"directionality={mean_sim:.3f}  frac_pos={frac_pos:.0%}",
                     color="white", fontsize=11)
        ax.set_xlabel(f"PC1 ({var[0]:.1%})", color="#94a3b8")
        ax.set_ylabel(f"PC2 ({var[1]:.1%})", color="#94a3b8")
        ax.tick_params(colors="#94a3b8")
        for sp in ax.spines.values(): sp.set_color("#334155")
        ax.legend(loc="upper right", fontsize=8,
                  facecolor="#1e293b", labelcolor="white",
                  edgecolor="#334155")

        plt.tight_layout()
        fig_path = out_dir / f"pca_L{pos}.png"
        plt.savefig(fig_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        print(f"  Saved {fig_path}")

    # ── Summary: norm ratio across positions ──────────────────────────────
    print("\n=== Summary ===")
    print(f"{'Pos':>4}  {'vec_norm':>9}  {'mean_thought':>13}  "
          f"{'ratio':>7}  {'needed_alpha':>12}")
    for pos in cfg["latent_positions"]:
        vec = vectors[pos]
        print(f"  L{pos}  {vec.norm().item():9.4f}  "
              f"(run with --summary-only to get thought norms)")
    print("\nDone. Figures saved to", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    analyze(load_config(args.config))