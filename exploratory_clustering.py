"""
src/exploratory_clustering.py

Exploratory signed-clustering analysis for the Korean discovery
corpus (paper §3.7, §5.3, Appendix B). This is explicitly
descriptive, not confirmatory -- the paper's primary result is the
orientation-coupling tests in src/orientation_tests.py.

Produces:
  - Primary clustering with k selected by silhouette
    (paper reports k=3 on the corrected N=71 corpus)
  - Full geometric-baseline ablation grid (paper Table "ablation")
  - GMM BIC model selection, Hopkins statistic, baseline--pre-event
    correlation (Appendix B, "Cluster tendency" / "Hierarchical
    structure" / "Feature correlation")
  - Diagnostic-category crosstab and chi-square test (Appendix
    Table B1)
  - Per-session spatial statistics for the two most divergent
    clusters (Appendix Table B2)
  - Sign-reflection robustness stress test (Appendix B,
    "Sign-Reflection Robustness"; also summarised in paper §3.6)

`find_best_k`, `cluster_at_k`, `permutation_test`, and
`bootstrap_stability` below are carried over largely unchanged
from an earlier version of this module -- they are feature-agnostic
(they standardize whatever array they are given) and did not
depend on the buggy Korean-corpus construction, so no correction
was needed for them specifically. Everything that touches session
selection or Lévy-area computation goes through
`compute_levy_features_window_v2` (src/preprocessing.py), the
corrected, bug-free pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                              normalized_mutual_info_score)
from scipy.stats import pearsonr, chi2_contingency, mannwhitneyu
from scipy.spatial import ConvexHull

from preprocessing import compute_levy_features_window_v2
from orientation_tests import (compute_levy_area_2d, precompute_coords,
                                features_from_cached_coords)


# ============================================================
# Carried over from the earlier clustering module (unchanged)
# ============================================================

def find_best_k(X, k_range=range(2, 7), n_init=10, random_state=42):
    """Selects k by maximising silhouette score. Returns (best_k, best_sil, all_scores)."""
    Xs = StandardScaler().fit_transform(X)
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = km.fit_predict(Xs)
        if len(np.unique(labels)) >= 2:
            scores[k] = silhouette_score(Xs, labels)
    best_k = max(scores, key=scores.get)
    return best_k, scores[best_k], scores


def cluster_at_k(X, k=4, n_init=10, random_state=42):
    """Clusters at a fixed k. Returns (labels, silhouette, per-cluster profiles)."""
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
    labels = km.fit_predict(Xs)
    sil = silhouette_score(Xs, labels)
    profiles = {}
    for c in range(k):
        mask = labels == c
        profiles[c] = {
            'n': int(mask.sum()),
            'base': float(X[mask, 0].mean()),
            'pre': float(X[mask, 1].mean()),
            'delta': float(X[mask, 2].mean()),
        }
    return labels, sil, profiles


def permutation_test(X, k, n_perm=1000, observed_sil=None, random_state=42):
    """Feature-wise permutation null for a k-means silhouette at fixed k."""
    Xs = StandardScaler().fit_transform(X)
    if observed_sil is None:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        observed_sil = silhouette_score(Xs, km.fit_predict(Xs))

    perm_sils = []
    for i in range(n_perm):
        rng = np.random.RandomState(i)
        Xp = Xs.copy()
        for col in range(Xs.shape[1]):
            rng.shuffle(Xp[:, col])
        km = KMeans(n_clusters=k, n_init=3, max_iter=100, random_state=i)
        labels = km.fit_predict(Xp)
        if len(np.unique(labels)) >= 2:
            perm_sils.append(silhouette_score(Xp, labels))

    perm_sils = np.array(perm_sils)
    p_val = np.mean(perm_sils >= observed_sil)
    return {'observed': observed_sil, 'null_mean': perm_sils.mean(),
            'null_std': perm_sils.std(), 'p_value': p_val,
            'null_distribution': perm_sils}


def bootstrap_stability(X, k, n_boot=1000, random_state=42):
    """Bootstrap resampling stability of the silhouette at fixed k."""
    from sklearn.utils import resample
    Xs = StandardScaler().fit_transform(X)
    boot_sils = []
    for i in range(n_boot):
        idx = resample(range(len(Xs)), random_state=i)
        Xb = Xs[idx]
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(Xb)
        if len(np.unique(labels)) == k:
            boot_sils.append(silhouette_score(Xb, labels))
    boot_sils = np.array(boot_sils)
    return {'mean': boot_sils.mean(),
            'ci_lower': np.percentile(boot_sils, 2.5),
            'ci_upper': np.percentile(boot_sils, 97.5),
            'distribution': boot_sils}


# ============================================================
# Hopkins statistic (cluster tendency)
# ============================================================

def hopkins_statistic(X, n_subsamples=None, random_state=42):
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.RandomState(random_state)
    n, d = X.shape
    m = min(n_subsamples or int(0.1 * n), n - 1)
    nbrs = NearestNeighbors(n_neighbors=2).fit(X)
    mins, maxs = X.min(axis=0), X.max(axis=0)

    idx = rng.choice(n, size=m, replace=False)
    u_dist, _ = nbrs.kneighbors(X[idx], n_neighbors=2)
    u_dist = u_dist[:, 1]

    X_random = rng.uniform(mins, maxs, size=(m, d))
    w_dist, _ = nbrs.kneighbors(X_random, n_neighbors=1)
    w_dist = w_dist[:, 0]

    return np.sum(w_dist) / (np.sum(w_dist) + np.sum(u_dist))


# ============================================================
# Geometric-baseline ablation
#
# *** Embedding position / Velocity / Curvature are
# RECONSTRUCTED from the Method-section spec (paper §4.2), not
# verbatim originals -- the exact original implementation could
# not be recovered. dLA-only and Base-only use columns of the
# canonical Lévy-area feature array directly and ARE exact. ***
# ============================================================

def _pre_event_window_coords(bal_features, sid, w=20, pre_turns=20):
    emb = np.array(bal_features[sid]['embeddings'])
    coords = PCA(n_components=2, random_state=42).fit_transform(emb)
    risk = bal_features[sid]['risk_turns']
    risk_indices = [i for i, r in enumerate(risk) if r == 1]
    if not risk_indices:
        return None
    risk_peaks, prev = [], -2
    for idx in risk_indices:
        if idx > prev + 1:
            risk_peaks.append(idx)
        prev = idx
    for peak in risk_peaks:
        if peak < pre_turns:
            continue
        lo = max(0, peak - pre_turns)
        if peak > lo:
            return coords[lo:peak]
    return None


def run_ablation_grid(bal_features, canonical_ids, X_levy, k_range=range(2, 6)):
    """
    Compares signed Lévy area against geometric-baseline feature
    variants (paper Table "ablation"). `X_levy` is the canonical
    [baseline, pre, diff] array from
    compute_levy_features_window_v2, in the same session order as
    `canonical_ids`.
    """
    feat_embedding, feat_velocity, feat_curvature = [], [], []
    valid_ids = []

    for sid in canonical_ids:
        pre_coords = _pre_event_window_coords(bal_features, sid)
        if pre_coords is None or len(pre_coords) < 3:
            continue

        emb_feat = [pre_coords[:, 0].mean(), pre_coords[:, 1].mean(),
                    pre_coords[:, 0].var(), pre_coords[:, 1].var()]

        diffs = np.diff(pre_coords, axis=0)
        speeds = np.linalg.norm(diffs, axis=1)
        vel_feat = [speeds.mean(), speeds.var()] if len(speeds) > 0 else [0, 0]

        if len(diffs) >= 2:
            angles = np.arctan2(diffs[:, 1], diffs[:, 0])
            dang = np.diff(angles)
            dang = (dang + np.pi) % (2 * np.pi) - np.pi
            curv_feat = [dang.mean(), dang.var()]
        else:
            curv_feat = [0, 0]

        feat_embedding.append(emb_feat)
        feat_velocity.append(vel_feat)
        feat_curvature.append(curv_feat)
        valid_ids.append(sid)

    mask = [sid in valid_ids for sid in canonical_ids]
    X_aligned = X_levy[mask]
    diff_col = X_aligned[:, 2]

    variants = {
        'Embedding position': np.array(feat_embedding),
        'Velocity (level-1)': np.array(feat_velocity),
        'Curvature': np.array(feat_curvature),
        'Unsigned |dLA|': np.abs(X_aligned),
        'Lévy area (ours)': X_aligned,
        'Lévy + Velocity': np.hstack([X_aligned, feat_velocity]),
        'dLA only': diff_col.reshape(-1, 1),
        'Base only': X_aligned[:, 0].reshape(-1, 1),
    }

    results = {}
    for name, feats in variants.items():
        feats_s = StandardScaler().fit_transform(feats)
        row = {}
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(feats_s)
            row[k] = (silhouette_score(feats_s, labels)
                       if len(np.unique(labels)) >= 2 else np.nan)
        results[name] = row

    # Random baseline
    sil_random = []
    for trial in range(1000):
        rng = np.random.RandomState(trial)
        random_feats = rng.permutation(X_aligned.flatten()).reshape(X_aligned.shape)
        random_feats_s = StandardScaler().fit_transform(random_feats)
        km = KMeans(n_clusters=4, random_state=42, n_init=3)
        labels = km.fit_predict(random_feats_s)
        if len(np.unique(labels)) >= 2:
            sil_random.append(silhouette_score(random_feats_s, labels))

    return results, (np.mean(sil_random), np.std(sil_random)), len(valid_ids)


# ============================================================
# Appendix Table B1 -- diagnostic category distribution
# ============================================================

def category_crosstab(df, cat_col='category', cluster_col='cluster'):
    cat_map = {'depression': 'D', 'anxiety': 'X', 'addiction': 'A', 'normative': 'N'}
    df = df.copy()
    df['cat_code'] = df[cat_col].map(cat_map)
    crosstab = pd.crosstab(df[cluster_col], df['cat_code']).reindex(
        columns=['D', 'X', 'A', 'N'], fill_value=0)
    chi2, p, dof, _ = chi2_contingency(crosstab)
    return crosstab, chi2, dof, p


# ============================================================
# Appendix Table B2 -- spatial statistics for the most divergent
# cluster pair
# ============================================================

def _path_length(pts):
    return np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))


def _end_start_dist(pts):
    return np.linalg.norm(pts[-1] - pts[0])


def _hull_area(pts):
    try:
        return ConvexHull(pts).volume
    except Exception:
        return np.nan


def spatial_statistics_table(bal_features, df, diff_col='diff', cluster_col='cluster'):
    """
    Identifies the two clusters with the most divergent mean
    Delta-LA and computes per-session first-order spatial metrics
    (hull area, end-start distance, path length) for each, with a
    Mann-Whitney U test per metric.
    """
    cluster_means = df.groupby(cluster_col)[diff_col].mean().sort_values()
    c_neg, c_pos = cluster_means.index[0], cluster_means.index[-1]

    def metrics_for(ids):
        rows = []
        for sid in ids:
            emb = np.array(bal_features[sid]['embeddings'])
            pts = PCA(n_components=2, random_state=42).fit_transform(emb)
            rows.append({'hull_area': _hull_area(pts),
                         'end_start_dist': _end_start_dist(pts),
                         'path_length': _path_length(pts)})
        return pd.DataFrame(rows)

    ids_neg = df.loc[df[cluster_col] == c_neg, 'dialogue_id'].tolist()
    ids_pos = df.loc[df[cluster_col] == c_pos, 'dialogue_id'].tolist()
    m_neg, m_pos = metrics_for(ids_neg), metrics_for(ids_pos)

    stats = {}
    for metric in ['hull_area', 'end_start_dist', 'path_length']:
        a, b = m_neg[metric].dropna(), m_pos[metric].dropna()
        u, p = mannwhitneyu(a, b)
        stats[metric] = {'mean_neg': a.mean(), 'std_neg': a.std(),
                          'mean_pos': b.mean(), 'std_pos': b.std(), 'p': p}

    return c_neg, c_pos, len(ids_neg), len(ids_pos), stats


# ============================================================
# Sign-reflection robustness (paper §3.6, Appendix "Sign-
# Reflection Robustness"). Uses the same feature pipeline as
# orientation_tests.py (precompute_coords / features_from_cached_
# coords) so the "unreflected" silhouette matches the primary
# clustering.
# ============================================================

def sign_reflection_stress_test(bal_features, canonical_ids, k, n_trials=200,
                                 flip_prob=0.5, random_state=42):
    cache = precompute_coords(bal_features, canonical_ids)
    rng = np.random.RandomState(random_state)

    rows_orig, valid_sids = [], []
    for sid in canonical_ids:
        d = cache.get(sid)
        if d is None:
            continue
        feat = features_from_cached_coords(d['coords'], d['risk'], flip_pc2=False)
        if feat:
            rows_orig.append(feat)
            valid_sids.append(sid)

    Xs_orig = StandardScaler().fit_transform(np.array(rows_orig))
    km_orig = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels_orig = km_orig.fit_predict(Xs_orig)
    sil_orig = silhouette_score(Xs_orig, labels_orig)

    aris, nmis, sils = [], [], []
    for trial in range(n_trials):
        rows = []
        for sid in valid_sids:
            d = cache[sid]
            flip = rng.rand() < flip_prob
            feat = features_from_cached_coords(d['coords'], d['risk'], flip_pc2=flip)
            rows.append(feat)
        Xs_flip = StandardScaler().fit_transform(np.array(rows))
        km_flip = KMeans(n_clusters=k, n_init=10, random_state=trial)
        labels_flip = km_flip.fit_predict(Xs_flip)

        aris.append(adjusted_rand_score(labels_orig, labels_flip))
        nmis.append(normalized_mutual_info_score(labels_orig, labels_flip))
        sils.append(silhouette_score(Xs_flip, labels_flip))

    return {
        'n': len(valid_sids), 'k': k, 'sil_orig': sil_orig,
        'sils': np.array(sils), 'aris': np.array(aris), 'nmis': np.array(nmis),
        'labels_orig': labels_orig,
    }


# ============================================================
# Reference values from the paper (for sanity-checking reruns)
# ============================================================

EXPECTED = {
    'N': 71,
    'best_k': 3,
    'silhouette_by_k': {2: .394, 3: .465, 4: .444, 5: .372},
    'cluster_sizes': {27, 28, 16},          # order/labels are arbitrary
    'gmm_bic_k': 2,
    'baseline_pre_r': .475,
    'hopkins_H': .801,
    'category_chi2': 8.23, 'category_p': .22,
    'sign_reflection_ari': .090, 'sign_reflection_nmi': .140,
    'sign_reflection_sil_mean': .414,
}


if __name__ == '__main__':
    import pickle

    with open('data/bal_features.pkl', 'rb') as f:
        bal_features = pickle.load(f)

    w20 = compute_levy_features_window_v2(bal_features, w=20, s=10,
                                           pre_turns=20, baseline_turns=30)
    canonical_ids = [r['dialogue_id'] for r in w20]
    assert len(canonical_ids) == EXPECTED['N'], (
        f"Expected N={EXPECTED['N']}, got {len(canonical_ids)} -- "
        "check src/preprocessing.py"
    )

    X71 = np.array([[r['baseline_levy'], r['pre_risk_levy'], r['diff']] for r in w20])

    print("=" * 60)
    print("Primary clustering (k selected by silhouette)")
    print("=" * 60)
    best_k, best_sil, all_scores = find_best_k(X71)
    for k, s in all_scores.items():
        print(f"  k={k}: silhouette={s:.4f}")
    print(f">>> best k = {best_k} (silhouette={best_sil:.4f}); "
          f"expected k={EXPECTED['best_k']}")

    labels, sil, profiles = cluster_at_k(X71, k=best_k)
    print("Cluster sizes:", {c: p['n'] for c, p in profiles.items()})

    df = pd.DataFrame(w20)
    df['category'] = df['dialogue_id'].map(lambda sid: bal_features[sid]['category'])
    df['cluster'] = labels

    print("\n" + "=" * 60)
    print("GMM BIC / correlation / Hopkins")
    print("=" * 60)
    Xs = StandardScaler().fit_transform(X71)
    bic_scores = [GaussianMixture(n_components=k, random_state=42, n_init=5)
                  .fit(Xs).bic(Xs) for k in range(2, 7)]
    print(f"BIC-selected k: {np.argmin(bic_scores) + 2} (expected {EXPECTED['gmm_bic_k']})")
    r, p = pearsonr(X71[:, 0], X71[:, 1])
    print(f"Baseline-pre correlation: r={r:.4f} (expected {EXPECTED['baseline_pre_r']})")
    H = np.mean([hopkins_statistic(Xs, n_subsamples=min(30, len(Xs) - 1),
                                    random_state=seed) for seed in range(100)])
    print(f"Hopkins H={H:.4f} (expected {EXPECTED['hopkins_H']})")

    print("\n" + "=" * 60)
    print("Appendix Table B1 -- category distribution")
    print("=" * 60)
    crosstab, chi2, dof, p = category_crosstab(df)
    print(crosstab)
    print(f"chi2={chi2:.2f}, dof={dof}, p={p:.2f} "
          f"(expected chi2={EXPECTED['category_chi2']}, p={EXPECTED['category_p']})")

    print("\n" + "=" * 60)
    print("Appendix Table B2 -- spatial statistics")
    print("=" * 60)
    c_neg, c_pos, n_neg, n_pos, stats = spatial_statistics_table(bal_features, df)
    print(f"Cluster {c_neg} (n={n_neg}) vs Cluster {c_pos} (n={n_pos})")
    for metric, s in stats.items():
        print(f"  {metric}: {s['mean_neg']:.2f}±{s['std_neg']:.2f} vs "
              f"{s['mean_pos']:.2f}±{s['std_pos']:.2f}  p={s['p']:.2f}")

    print("\n" + "=" * 60)
    print("Sign-reflection robustness")
    print("=" * 60)
    sr = sign_reflection_stress_test(bal_features, canonical_ids, k=best_k)
    print(f"Unreflected silhouette: {sr['sil_orig']:.4f}")
    print(f"Mean silhouette across flips: {sr['sils'].mean():.4f} \u00b1 {sr['sils'].std():.4f}")
    print(f"ARI: {sr['aris'].mean():.4f} \u00b1 {sr['aris'].std():.4f} "
          f"(expected ~{EXPECTED['sign_reflection_ari']})")
    print(f"NMI: {sr['nmis'].mean():.4f} \u00b1 {sr['nmis'].std():.4f} "
          f"(expected ~{EXPECTED['sign_reflection_nmi']})")

    print("\n" + "=" * 60)
    print("Ablation grid")
    print("=" * 60)
    ablation, random_baseline, n_used = run_ablation_grid(bal_features, canonical_ids, X71)
    print(f"N used: {n_used}")
    for name, row in ablation.items():
        best_kv = max(row, key=row.get)
        print(f"  {name:22s}: " + " ".join(f"k={k}:{v:.3f}" for k, v in row.items()) +
              f"  | best k={best_kv} ({row[best_kv]:.3f})")
    print(f"Random baseline: {random_baseline[0]:.3f} \u00b1 {random_baseline[1]:.3f}")
