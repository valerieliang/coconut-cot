"""
analyze_steering_vectors.py
"""

import argparse, json, os, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml
from transformers import GPT2LMHeadModel, GPT2Tokenizer

LATENT_TOKEN_ID = 50257
START_LATENT_ID = 50258
END_LATENT_ID   = 50259
EOS_TOKEN_ID    = 50256
VOCAB_SIZE      = 50260

BG       = "#ffffff"
CORRECT  = "#16a34a"
WRONG    = "#dc2626"
C_TEST   = "#15803d"
W_TEST   = "#b91c1c"
PAIR_CLR = "#94a3b8"
VEC_MEAN = "#d97706"
VEC_SV   = "#b45309"
MUTED    = "#cbd5e1"
GRID_CLR = "#f1f5f9"


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
    from coconut.coconut import Coconut
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


def _center_and_scale(proj):
    center = proj.mean(axis=0)
    proj = proj - center
    p_low, p_high = np.percentile(proj, [5, 95], axis=0)
    scale = np.mean(p_high - p_low) / 2 + 1e-8
    return proj / scale


def _confidence_ellipse(x, y, ax, n_std=1.5, **kwargs):
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    ax.add_patch(Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=w, height=h, angle=theta, **kwargs
    ))


def _draw_word2vec_arrows(ax, wrong_proj, correct_proj, sv_proj):
    wrong_c   = wrong_proj.mean(axis=0)
    correct_c = correct_proj.mean(axis=0)

    ax.annotate("", xy=correct_c, xytext=wrong_c,
        arrowprops=dict(arrowstyle="-|>", color=VEC_MEAN, lw=2.4,
                        mutation_scale=18), zorder=10)

    sv_dir = sv_proj - wrong_c
    sv_norm = np.linalg.norm(sv_dir) + 1e-8
    mean_diff_len = np.linalg.norm(correct_c - wrong_c)
    sv_tip = wrong_c + sv_dir / sv_norm * mean_diff_len
    ax.annotate("", xy=sv_tip, xytext=wrong_c,
        arrowprops=dict(arrowstyle="-|>", color=VEC_SV, lw=1.5,
                        linestyle="dashed", mutation_scale=14), zorder=9)

    for wpt, cpt in zip(wrong_proj, correct_proj):
        ax.annotate("", xy=cpt, xytext=wpt,
            arrowprops=dict(arrowstyle="-|>", color=PAIR_CLR, lw=0.6,
                            mutation_scale=8, alpha=0.35), zorder=5)

    return wrong_c, correct_c, sv_tip


def plot_pca(pos, proj_all, n_test, cos_sim_mean, frac_pos, var, out_dir):
    proj_all_norm = _center_and_scale(proj_all.copy())

    p_correct = proj_all_norm[:n_test]
    p_wrong   = proj_all_norm[n_test:2*n_test]
    p_sv      = proj_all_norm[-1]

    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    all_pts = proj_all_norm[:-1]
    low, high = np.percentile(all_pts, [2, 98])
    lim = max(abs(low), abs(high))
    margin = 0.15 * lim
    ax.set_xlim(-lim - margin, lim + margin)
    ax.set_ylim(-lim - margin, lim + margin)

    ax.grid(True, color=GRID_CLR, linewidth=0.5)
    ax.axhline(0, color=MUTED, lw=0.5)
    ax.axvline(0, color=MUTED, lw=0.5)

    _confidence_ellipse(p_correct[:,0], p_correct[:,1], ax,
        facecolor=CORRECT, alpha=0.08, edgecolor=CORRECT, linewidth=1.0,
        linestyle="--")
    _confidence_ellipse(p_wrong[:,0], p_wrong[:,1], ax,
        facecolor=WRONG, alpha=0.08, edgecolor=WRONG, linewidth=1.0,
        linestyle="--")

    ax.scatter(p_correct[:,0], p_correct[:,1],
               c=C_TEST, s=45, marker="^", alpha=0.9, label="correct", zorder=4)
    ax.scatter(p_wrong[:,0], p_wrong[:,1],
               c=W_TEST, s=45, marker="^", alpha=0.9, label="wrong", zorder=4)

    wrong_c, correct_c, sv_tip = _draw_word2vec_arrows(ax, p_wrong, p_correct, p_sv)

    ax.scatter(*wrong_c,   c=WRONG,   s=90, marker="x", lw=2, zorder=11)
    ax.scatter(*correct_c, c=CORRECT, s=90, marker="x", lw=2, zorder=11)

    mid = (wrong_c + correct_c) / 2
    ax.annotate(f"mean(c−w)  cos={cos_sim_mean:.2f}", xy=mid,
        xytext=(mid[0]+0.15, mid[1]+0.12), fontsize=8, color=VEC_MEAN,
        arrowprops=dict(arrowstyle="-", color=VEC_MEAN, lw=0.6))
    mid_sv = (wrong_c + sv_tip) / 2
    ax.annotate("steering vec", xy=mid_sv,
        xytext=(mid_sv[0]-0.28, mid_sv[1]-0.16), fontsize=8, color=VEC_SV,
        arrowprops=dict(arrowstyle="-", color=VEC_SV, lw=0.6))

    ax.set_title(
        f"L{pos} | PC1={var[0]:.1%} PC2={var[1]:.1%} "
        f"dir={cos_sim_mean:.3f} frac={frac_pos:.0%}",
        color="black")
    ax.set_xlabel(f"PC1 ({var[0]:.1%})", color="#475569")
    ax.set_ylabel(f"PC2 ({var[1]:.1%})", color="#475569")
    ax.tick_params(colors="#475569")
    for sp in ax.spines.values(): sp.set_color("#e2e8f0")
    ax.legend(loc="upper right", fontsize=8,
              facecolor="#f8fafc", labelcolor="black", edgecolor="#e2e8f0")

    plt.tight_layout()
    fig_path = Path(os.path.expanduser(str(out_dir))) / f"pca_L{pos}.png"
    plt.savefig(fig_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved {fig_path}")


def analyze(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model     = load_model(cfg, device)
    tokenizer = load_tokenizer()
    n_latent  = 6

    vectors = torch.load(
        os.path.expanduser(cfg["output_dir"]) + "/steering_vectors.pt",
        map_location="cpu",
    )

    with open(os.path.expanduser(cfg["train_path"])) as f:
        all_data = json.load(f)

    n_train   = cfg["n_contrast"]
    test_data = all_data[n_train: n_train + 200]

    out_dir = Path(os.path.expanduser(cfg["output_dir"])) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    for pos in cfg["latent_positions"]:
        capture_pass = pos - 1
        vec = vectors[pos]

        print(f"\n=== L{pos} ===")

        correct_test, wrong_test = [], []
        for s in test_data[:100]:
            with torch.no_grad():
                hc = get_thought(model, tokenizer, s["question"],
                                 n_latent, device, capture_pass)
                hw = get_thought(model, tokenizer, build_neg_question(s),
                                 n_latent, device, capture_pass)
            if hc is not None: correct_test.append(hc)
            if hw is not None: wrong_test.append(hw)

        diffs = [c - w for c, w in zip(correct_test, wrong_test)]
        sims  = [cosine_sim(d, vec) for d in diffs]

        mat  = torch.stack(correct_test + wrong_test + [vec]).float().numpy()
        pca  = PCA(n_components=2)
        proj = pca.fit_transform(mat)

        plot_pca(
            pos=pos,
            proj_all=proj,
            n_test=len(correct_test),
            cos_sim_mean=np.mean(sims),
            frac_pos=np.mean([s > 0 for s in sims]),
            var=pca.explained_variance_ratio_,
            out_dir=out_dir,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    analyze(load_config(args.config))