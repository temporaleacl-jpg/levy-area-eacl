"""
figures/appendix_dendrogram.py

Appendix Figure A1: exploratory cluster structure of signed
Lévy-area features in the corrected Korean corpus (N=71, k=3
silhouette-selected; paper Appendix B, "Cluster tendency" /
"Hierarchical structure").

Reuses the canonical, bug-free feature pipeline from
src/preprocessing.py rather than reading from any pre-computed
cache, so this script cannot silently regenerate the earlier N=70
figure (see src/preprocessing.py module docstring for the A105
indexing-bug background).
"""

import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

from preprocessing import compute_levy_features_window_v2

plt.rcParams.update({'font.family': 'serif',
                      'mathtext.fontset': 'stix',
                      'font.size': 10})

CLUSTER_COLORS = {0: '#5B9FD4', 1: '#C0392B', 2: '#185FA5'}
MARKERS = {0: 'o', 1: '^', 2: 's'}

EXPECTED_N = 71
EXPECTED_K = 3
EXPECTED_SIZES = {27, 28, 16}


def build_appendix_figure(bal_features, out_path='figA1_final.png'):
    w20 = compute_levy_features_window_v2(bal_features, w=20, s=10,
                                           pre_turns=20, baseline_turns=30)
    canonical_ids = [r['dialogue_id'] for r in w20]

    print(f"N = {len(canonical_ids)} (expected {EXPECTED_N})")
    assert len(canonical_ids) == EXPECTED_N, (
        f"Expected N={EXPECTED_N} (corrected Korean corpus, including "
        f"dialogue ID A105); got N={len(canonical_ids)}. See "
        "src/preprocessing.py module docstring."
    )
    assert 'A105' in canonical_ids, (
        "A105 missing -- this looks like the buggy pre-corrected pipeline."
    )

    X = np.array([[r['baseline_levy'], r['pre_risk_levy'], r['diff']] for r in w20])
    X_scaled = StandardScaler().fit_transform(X)

    # ── Primary clustering (k=3, silhouette-selected on the corrected corpus) ──
    kmeans = KMeans(n_clusters=EXPECTED_K, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    sizes = dict(zip(*np.unique(labels, return_counts=True)))
    print(f"클러스터 분포: {sizes}")
    if set(sizes.values()) != EXPECTED_SIZES:
        print(f"⚠️  WARNING: cluster sizes {set(sizes.values())} do not match "
              f"expected {EXPECTED_SIZES} -- do not treat this as the "
              "validated discovery-corpus solution until resolved.")
    else:
        print("✅ Cluster sizes match the corrected Table B1.")

    # ── GMM (BIC-selected k) for the coarse two-cluster overlay ──
    bic_scores = [GaussianMixture(n_components=k, random_state=42, n_init=5)
                  .fit(X_scaled).bic(X_scaled) for k in range(2, 7)]
    best_k_bic = int(np.argmin(bic_scores)) + 2
    print(f"BIC-selected k: {best_k_bic} (scores: "
          f"{[f'{b:.1f}' for b in bic_scores]})")
    gmm2 = GaussianMixture(n_components=2, random_state=42, n_init=10)
    labels_gmm2 = gmm2.fit_predict(X_scaled)

    # ── Plot ──
    Z = linkage(X_scaled, method='ward')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel (a): dendrogram
    ax = axes[0]
    dend = dendrogram(Z, ax=ax, no_labels=True, color_threshold=0,
                       above_threshold_color='#AAAAAA')

    heights = sorted(Z[:, 2], reverse=True)
    cut2 = (heights[1] + heights[2]) / 2
    cut3 = (heights[2] + heights[3]) / 2
    ax.axhline(cut2, color='#C0392B', linewidth=1.8, linestyle='--',
               label='k=2 cut', zorder=5)
    ax.axhline(cut3, color='#1D9E75', linewidth=1.8, linestyle='--',
               label='k=3 cut', zorder=5)

    leaf_order = dend['leaves']
    n = len(leaf_order)
    bar_w = (ax.get_xlim()[1] - ax.get_xlim()[0]) / n
    for i, leaf_idx in enumerate(leaf_order):
        color = CLUSTER_COLORS[labels[leaf_idx]]
        ax.add_patch(plt.Rectangle(
            (ax.get_xlim()[0] + i * bar_w, -1.2), bar_w, 0.8,
            color=color, clip_on=False))

    ax.set_title('(a) Hierarchical clustering\n(Ward linkage)',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Sessions', fontsize=10)
    ax.set_ylabel('Distance', fontsize=10)
    ax.set_ylim(-1.5, None)
    ax.legend(fontsize=9, loc='upper left')
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    ax.set_xticks([])

    # Panel (b): baseline vs pre-event scatter
    ax2 = axes[1]
    for g, fc in [(0, '#185FA5'), (1, '#C0392B')]:
        mask = labels_gmm2 == g
        ax2.scatter(X_scaled[mask, 0], X_scaled[mask, 1], c=fc, alpha=0.10,
                    s=200, edgecolors='none', zorder=1)

    for k in range(EXPECTED_K):
        mask = labels == k
        ax2.scatter(X_scaled[mask, 0], X_scaled[mask, 1],
                    c=CLUSTER_COLORS[k], marker=MARKERS[k], s=60, alpha=0.85,
                    edgecolors='white', linewidths=0.6,
                    label=f'Cluster {k} (n={mask.sum()})', zorder=3)

    ax2.axvline(0, color='#ccc', linewidth=0.8, linestyle=':', zorder=0)
    ax2.axhline(0, color='#ccc', linewidth=0.8, linestyle=':', zorder=0)
    ax2.plot([-3, 3], [-3, 3], color='#ccc', linewidth=0.6, linestyle='-',
              alpha=0.5, zorder=0)
    ax2.annotate('no change\n(base=pre)', xy=(1.7, 1.7), fontsize=7.5,
                 color='#999', rotation=45, ha='center')

    ax2.set_xlabel('Baseline LA (z-scored)', fontsize=10)
    ax2.set_ylabel('Pre-event LA (z-scored)', fontsize=10)
    ax2.set_title('(b) Cluster structure\n'
                  '(background = GMM k=2 groups, BIC-selected)',
                  fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8, loc='lower right', framealpha=0.92)
    ax2.spines[['top', 'right']].set_visible(False)

    sil_gmm2 = silhouette_score(X_scaled, labels_gmm2)
    sil_km = silhouette_score(X_scaled, labels)
    ax2.text(0.03, 0.96,
             f'sil(k=2, GMM)={sil_gmm2:.3f}\nsil(k={EXPECTED_K}, k-means)={sil_km:.3f}',
             transform=ax2.transAxes, ha='left', va='top', fontsize=8.5,
             color='#444', bbox=dict(boxstyle='round,pad=0.3', fc='white',
                                      ec='#ccc', alpha=0.9))

    plt.suptitle(f'Exploratory cluster structure of signed\n'
                 f'Lévy-area features (Korean corpus, N={EXPECTED_N})',
                 fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    return fig, axes


if __name__ == '__main__':
    with open('data/bal_features.pkl', 'rb') as f:
        bal_features = pickle.load(f)
    build_appendix_figure(bal_features)
    print("완료!")
