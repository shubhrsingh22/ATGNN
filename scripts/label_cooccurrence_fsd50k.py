"""Label frequency and co-occurrence analysis for FSD50K.

Motivates the ATGNN label graphs: the PLG (patch-label) and LLG (label-label)
blocks assume that (a) clips are genuinely multi-label and (b) labels co-occur
with strong, structured correlations that a learnable label graph can exploit.

Computes from the official ground truth (dev.csv + eval.csv):
  - labels-per-clip histogram
  - per-class frequency (long-tail plot)
  - label co-occurrence: counts C[i, j], conditional probabilities
    P(j | i) = C[i, j] / C[i, i], and normalised pointwise mutual information
  - top co-occurring label pairs

Outputs (default under results/):
  label_cooccurrence.png    4-panel summary figure
  label_stats.csv           per-class frequencies
  top_label_pairs.csv       top-100 pairs by count with P(j|i) and nPMI

Usage:
    python scripts/label_cooccurrence_fsd50k.py \
        --gt_dir /path/to/FSD50K/ground_truth --out_dir results
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_labels(gt_dir):
    vocab = pd.read_csv(os.path.join(gt_dir, "vocabulary.csv"),
                        header=None, names=["index", "name", "mid"])
    names = vocab["name"].tolist()
    name_to_idx = {n: i for i, n in enumerate(names)}

    frames = []
    for f in ("dev.csv", "eval.csv"):
        df = pd.read_csv(os.path.join(gt_dir, f))
        frames.append(df[["fname", "labels"]])
    df = pd.concat(frames, ignore_index=True)

    S = len(names)
    Y = np.zeros((len(df), S), dtype=np.float32)
    for r, labs in enumerate(df["labels"]):
        for lab in labs.split(","):
            Y[r, name_to_idx[lab]] = 1.0
    return Y, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--top_k_heatmap", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    Y, names = load_labels(args.gt_dir)
    n_clips, S = Y.shape
    counts = Y.sum(0)                      # per-class clip counts
    labels_per_clip = Y.sum(1)
    C = Y.T @ Y                            # co-occurrence counts (S x S)

    # conditional probability P(j | i)
    P = C / np.maximum(counts[:, None], 1)
    np.fill_diagonal(P, 0)

    # normalised PMI over label pairs
    p_i = counts / n_clips
    p_ij = C / n_clips
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(p_ij / (p_i[:, None] * p_i[None, :]))
        npmi = pmi / (-np.log(p_ij))
    npmi[~np.isfinite(npmi)] = 0
    np.fill_diagonal(npmi, 0)

    multi = (labels_per_clip > 1).mean() * 100
    print(f"clips: {n_clips}, classes: {S}")
    print(f"labels/clip: mean {labels_per_clip.mean():.2f}, median {np.median(labels_per_clip):.0f}, "
          f"max {labels_per_clip.max():.0f}; multi-label clips: {multi:.1f}%")

    # per-class stats CSV
    pd.DataFrame({"label": names, "clips": counts.astype(int),
                  "fraction": counts / n_clips}).sort_values(
        "clips", ascending=False).to_csv(
        os.path.join(args.out_dir, "label_stats.csv"), index=False)

    # top pairs CSV (upper triangle by count)
    iu = np.triu_indices(S, k=1)
    order = np.argsort(C[iu])[::-1][:100]
    rows = []
    for o in order:
        i, j = iu[0][o], iu[1][o]
        rows.append(dict(label_i=names[i], label_j=names[j],
                         count=int(C[i, j]),
                         p_j_given_i=round(P[i, j], 3),
                         p_i_given_j=round(P[j, i], 3),
                         npmi=round(npmi[i, j], 3)))
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "top_label_pairs.csv"), index=False)
    print("top pairs:", [(r['label_i'], r['label_j'], r['count']) for r in rows[:5]])

    # ---- figure ----
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    ax = axes[0, 0]
    ax.hist(labels_per_clip, bins=np.arange(0.5, labels_per_clip.max() + 1.5), color="#4878cf")
    ax.set_xlabel("labels per clip")
    ax.set_ylabel("clips")
    ax.set_title(f"Labels per clip (mean {labels_per_clip.mean():.2f}; "
                 f"{multi:.0f}% of clips are multi-label)")

    ax = axes[0, 1]
    ax.plot(np.sort(counts)[::-1], color="#4878cf")
    ax.set_yscale("log")
    ax.set_xlabel("class rank")
    ax.set_ylabel("clips (log)")
    ax.set_title("Class frequency (long tail)")

    # heatmap of P(j|i) for the most frequent classes
    k = args.top_k_heatmap
    top = np.argsort(counts)[::-1][:k]
    ax = axes[1, 0]
    im = ax.imshow(P[np.ix_(top, top)], cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    short = [names[t][:18] for t in top]
    ax.set_xticklabels(short, rotation=90, fontsize=6)
    ax.set_yticklabels(short, fontsize=6)
    ax.set_title(f"P(column | row), top-{k} classes")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # distribution of strongest conditional per class
    ax = axes[1, 1]
    ax.hist(P.max(1), bins=40, color="#4878cf")
    ax.set_xlabel("max_j P(j | i)")
    ax.set_ylabel("classes")
    strong = (P.max(1) > 0.5).sum()
    ax.set_title(f"Strongest co-occurrence per class "
                 f"({strong}/{S} classes have max P > 0.5)")

    fig.suptitle("FSD50K label statistics (dev + eval ground truth)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_png = os.path.join(args.out_dir, "label_cooccurrence.png")
    fig.savefig(out_png, dpi=150)
    print(f"figure -> {out_png}")


if __name__ == "__main__":
    main()
