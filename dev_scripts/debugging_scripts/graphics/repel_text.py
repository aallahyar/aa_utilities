"""Volcano plot demo for repel_text."""

import numpy as np
import matplotlib.pyplot as plt
from aa_utilities.graphics import repel_text

# ---------------------------------------------------------------------------
# Synthetic volcano-plot data
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
n_genes = 5000

log2fc = rng.normal(0, 1.5, n_genes)
neg_log10p = rng.exponential(1.0, n_genes)

# Make a handful of genes clearly significant (large |fc| + high -log10p).
n_sig = 50
sig_idx = rng.choice(n_genes, n_sig, replace=False)
log2fc[sig_idx] = rng.choice([-1, 1], n_sig) * rng.uniform(2, 5, n_sig)
neg_log10p[sig_idx] = rng.uniform(5, 15, n_sig)

gene_names = [f"Gene{i}" for i in range(n_genes)]

# Thresholds
fc_thresh = 1.5
p_thresh = 3.0
is_sig = (np.abs(log2fc) > fc_thresh) & (neg_log10p > p_thresh)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 7))

sc1 = ax.scatter(log2fc[~is_sig], neg_log10p[~is_sig], c="0.75", s=6, alpha=0.5,
                 edgecolors="none", label="_nolegend_")

colors = np.where(log2fc[is_sig] > 0, "tab:red", "tab:blue")
sc2 = ax.scatter(log2fc[is_sig], neg_log10p[is_sig], c=colors, s=18, alpha=0.8,
                 edgecolors="none")

# Threshold lines
ax.axhline(p_thresh, ls="--", lw=0.5, color="0.6")
ax.axvline(-fc_thresh, ls="--", lw=0.5, color="0.6")
ax.axvline(fc_thresh, ls="--", lw=0.5, color="0.6")

ax.set_xlabel("log₂ fold change")
ax.set_ylabel("−log₁₀ p-value")
ax.set_title("Volcano plot — repel_text demo")

# ---------------------------------------------------------------------------
# Label the top 25 genes (by significance, then fold-change)
# ---------------------------------------------------------------------------
top = np.where(is_sig)[0]
order = np.lexsort((-np.abs(log2fc[top]), -neg_log10p[top]))
top = top[order][:50]

repel_text(
    ax,
    log2fc[top],
    neg_log10p[top],
    [gene_names[i] for i in top],
    avoid=[sc1, sc2],
    text_kwargs=dict(fontsize=8),
    arrow_kwargs=dict(color="0.4", lw=0.5),
    min_distance=8,
    max_distance=100,
    margin=4,
)

plt.tight_layout()
# plt.savefig("volcano_demo.png", dpi=150)
plt.show()
