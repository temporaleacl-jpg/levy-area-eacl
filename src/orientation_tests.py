"""
src/orientation_tests.py

The two confirmatory statistical tests underlying the paper's
primary result (paper §3.5, §5.1-5.2):

  1. `orientation_coupling_test` -- pairing permutation. Tests
     whether baseline--pre-event orientation coupling exceeds what
     would be expected if the two windows were independent across
     sessions (Table "pairing" in the paper).

  2. `matched_pseudo_anchor_test*` -- matched pseudo-anchor test.
     Tests whether that coupling is specifically enriched near the
     true transition, by comparing the true-anchor same-orientation
     rate against a null built from pseudo-anchors sampled within
     the same sessions (Table 1 "hero" in the paper; full values in
     Appendix A).

Three corpus-specific variants of the pseudo-anchor test are
provided because each corpus's session representation differs
(Korean: a list of risk-flagged turns per session; Supreme Court:
a single pre-filtered advocate-only turn sequence with a scalar
anchor). `matched_pseudo_anchor_test_generic` unifies these for
ESConv and ANNO-MI, whose sessions -- like Supreme Court's --
carry a single scalar anchor turn per session.
"""

import time

import numpy as np
from sklearn.decomposition import PCA

from levy_area import compute_levy_area_2d


# ============================================================
# 1. Pairing permutation test (generic across corpora)
# ============================================================

def orientation_coupling_test(base_pre_pairs, n_perm=5000,
                               random_state=42, corpus_name=""):
    """
    base_pre_pairs: list of (baseline_levy, pre_event_levy) tuples,
    one per session.

    Permutes the pairing between baseline and pre-event values
    across sessions (preserving each variable's marginal
    distribution) to build a null distribution for the observed
    same-orientation rate T = mean(sign(base) == sign(pre)).
    """
    base = np.array([p[0] for p in base_pre_pairs])
    pre = np.array([p[1] for p in base_pre_pairs])
    N = len(base)

    observed_T = np.mean(np.sign(base) == np.sign(pre))

    rng = np.random.RandomState(random_state)
    null_Ts = np.array([
        np.mean(np.sign(base) == np.sign(rng.permutation(pre)))
        for _ in range(n_perm)
    ])
    p_val = np.mean(null_Ts >= observed_T)

    # Bootstrap CI for the observed T (session resampling)
    boot_Ts = []
    for _ in range(1000):
        idx = rng.randint(0, N, size=N)
        boot_Ts.append(np.mean(np.sign(base[idx]) == np.sign(pre[idx])))
    ci_lower, ci_upper = np.percentile(boot_Ts, [2.5, 97.5])

    result = {
        'corpus': corpus_name, 'N': N, 'observed_T': observed_T,
        'null_mean': null_Ts.mean(), 'null_std': null_Ts.std(),
        'p_value': p_val, 'ci_lower': ci_lower, 'ci_upper': ci_upper,
    }
    print(f"[{corpus_name}] N={N}, T={observed_T:.3f} "
          f"[{ci_lower:.3f}, {ci_upper:.3f}], "
          f"null={null_Ts.mean():.3f}\u00b1{null_Ts.std():.3f}, p={p_val:.4f}")
    return result


# ============================================================
# 2a. Matched pseudo-anchor test -- Korean counseling
#     (session = list of risk-flagged turns; see
#     src/preprocessing.py for `bal_features` / `pre_results`)
# ============================================================

def precompute_coords(bal_features, sids):
    """Fits session-wise PCA once per session and caches the result."""
    cache = {}
    for sid in sids:
        d = bal_features.get(sid)
        if d is None:
            continue
        emb = np.array(d['embeddings'])
        pca = PCA(n_components=2)
        coords = pca.fit_transform(emb)
        cache[sid] = {'coords': coords, 'risk': d['risk_turns']}
    return cache


def features_from_cached_coords(coords, risk, flip_pc2=False,
                                 w=20, s=10, pre_turns=20,
                                 baseline_turns=30):
    """
    Computes [baseline_levy, pre_event_levy, diff] from cached
    session coordinates. `flip_pc2=True` reflects the second PCA
    axis without recomputing PCA -- used only by the sign-reflection
    robustness check (src/exploratory_clustering.py), not here.
    """
    c = coords.copy()
    if flip_pc2:
        c[:, 1] *= -1

    la_seq, la_turn_start = [], []
    for start in range(0, len(c) - w + 1, s):
        la_seq.append(compute_levy_area_2d(c[start:start + w]))
        la_turn_start.append(start)
    if len(la_seq) < 4:
        return None

    la_seq = np.array(la_seq)
    if np.std(la_seq) > 1e-10:
        la_seq = (la_seq - np.mean(la_seq)) / np.std(la_seq)

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
        pre_mask = [(t >= peak - pre_turns) and (t < peak) for t in la_turn_start]
        base_mask = [t < baseline_turns for t in la_turn_start]
        if sum(pre_mask) == 0 or sum(base_mask) == 0:
            continue
        pre_levy = np.mean(la_seq[pre_mask])
        baseline_levy = np.mean(la_seq[base_mask])
        return [baseline_levy, pre_levy, pre_levy - baseline_levy]
    return None


def matched_pseudo_anchor_test(bal_features, pre_results, n_trials=1000,
                                w=20, s=10, pre_turns=20, baseline_turns=30,
                                random_state=42):
    """
    Korean-counseling matched pseudo-anchor test (paper Table 1,
    Korean row; N=71 with the corrected corpus).
    """
    sids = [r['dialogue_id'] for r in pre_results]
    cache = precompute_coords(bal_features, sids)
    rng = np.random.RandomState(random_state)

    real_pairs = []
    for sid, d in cache.items():
        feat = features_from_cached_coords(d['coords'], d['risk'], flip_pc2=False,
                                            w=w, s=s, pre_turns=pre_turns,
                                            baseline_turns=baseline_turns)
        if feat:
            real_pairs.append((feat[0], feat[1]))
    real_T = np.mean([np.sign(b) == np.sign(p) for b, p in real_pairs])
    print(f"실제 anchor 기준 T = {real_T:.4f} (N={len(real_pairs)})")

    pseudo_Ts = []
    for trial in range(n_trials):
        pairs = []
        for sid, d in cache.items():
            coords = d['coords']
            la_seq, la_turn_start = [], []
            for start in range(0, len(coords) - w + 1, s):
                la_seq.append(compute_levy_area_2d(coords[start:start + w]))
                la_turn_start.append(start)
            if len(la_seq) < 4:
                continue
            la_seq = np.array(la_seq)
            if np.std(la_seq) > 1e-10:
                la_seq = (la_seq - np.mean(la_seq)) / np.std(la_seq)

            T_total = len(coords)
            if T_total < pre_turns + 10:
                continue
            pseudo_peak = rng.randint(pre_turns, T_total)

            pre_mask = [(t >= pseudo_peak - pre_turns) and (t < pseudo_peak)
                        for t in la_turn_start]
            base_mask = [t < baseline_turns for t in la_turn_start]
            if sum(pre_mask) == 0 or sum(base_mask) == 0:
                continue

            pre_levy = np.mean(la_seq[pre_mask])
            baseline_levy = np.mean(la_seq[base_mask])
            pairs.append((baseline_levy, pre_levy))

        if len(pairs) < 10:
            continue
        T = np.mean([np.sign(b) == np.sign(p) for b, p in pairs])
        pseudo_Ts.append(T)

    pseudo_Ts = np.array(pseudo_Ts)
    p_val = np.mean(pseudo_Ts >= real_T)
    print(f"\nPseudo-anchor 분포 (n={len(pseudo_Ts)} trials): "
          f"mean={pseudo_Ts.mean():.4f} \u00b1 {pseudo_Ts.std():.4f}")
    print(f"p-value (real >= pseudo): {p_val:.4f}")
    return real_T, pseudo_Ts, p_val


# ============================================================
# 2b. Matched pseudo-anchor test -- Supreme Court
#     (session = pre-filtered advocate-only utterance list with a
#     single scalar anchor turn; see
#     src/preprocessing.filter_valid_sessions_advocate_only)
# ============================================================

def matched_pseudo_anchor_test_supreme(valid_sessions_adv, labse,
                                        n_trials=1000, w=10, s=5,
                                        pre_turns=20, baseline_turns=15,
                                        max_sessions=2000,
                                        batch_size=256,
                                        random_state=42):
    """
    Supreme Court matched pseudo-anchor test (paper Table 1, Supreme
    Court row; N=1,998 of the first 2,000 advocate-only anchorable
    sessions, capped for computational tractability -- see paper
    Limitations).
    """
    sessions_use = valid_sessions_adv[:max_sessions]
    print(f"사용 세션 수: {len(sessions_use)} (cap={max_sessions})")

    all_texts, boundaries = [], []
    for sess in sessions_use:
        start = len(all_texts)
        all_texts.extend(sess['texts'])
        boundaries.append((start, len(all_texts)))
    print(f"전체 발화 수: {len(all_texts)}")

    t0 = time.time()
    all_emb = labse.encode(all_texts, batch_size=batch_size,
                            show_progress_bar=True, convert_to_numpy=True)
    print(f"인코딩 완료: {time.time() - t0:.1f}s")

    coords_cache, anchors = {}, {}
    for i, (sess, (start, end)) in enumerate(zip(sessions_use, boundaries)):
        emb = all_emb[start:end]
        if len(emb) < w:
            continue
        pca = PCA(n_components=2)
        coords_cache[i] = pca.fit_transform(emb)
        anchors[i] = sess['anchor']
    print(f"PCA 캐싱 완료, 유효 세션: {len(coords_cache)}")

    real_pairs = []
    for i, coords in coords_cache.items():
        la_seq, la_turn_start = [], []
        for start in range(0, len(coords) - w + 1, s):
            la_seq.append(compute_levy_area_2d(coords[start:start + w]))
            la_turn_start.append(start)
        if len(la_seq) < 4:
            continue
        la_seq = np.array(la_seq)
        if np.std(la_seq) > 1e-10:
            la_seq = (la_seq - np.mean(la_seq)) / np.std(la_seq)

        anchor = anchors[i]
        if anchor < pre_turns:
            continue
        pre_mask = [(t >= anchor - pre_turns) and (t < anchor) for t in la_turn_start]
        base_mask = [t < baseline_turns for t in la_turn_start]
        if sum(pre_mask) == 0 or sum(base_mask) == 0:
            continue

        pre_levy = np.mean(la_seq[pre_mask])
        baseline_levy = np.mean(la_seq[base_mask])
        real_pairs.append((baseline_levy, pre_levy))

    real_T = np.mean([np.sign(b) == np.sign(p) for b, p in real_pairs])
    print(f"실제 anchor 기준 T = {real_T:.4f} (N={len(real_pairs)})")

    rng = np.random.RandomState(random_state)
    pseudo_Ts = []
    t1 = time.time()
    for trial in range(n_trials):
        pairs = []
        for i, coords in coords_cache.items():
            la_seq, la_turn_start = [], []
            for start in range(0, len(coords) - w + 1, s):
                la_seq.append(compute_levy_area_2d(coords[start:start + w]))
                la_turn_start.append(start)
            if len(la_seq) < 4:
                continue
            la_seq = np.array(la_seq)
            if np.std(la_seq) > 1e-10:
                la_seq = (la_seq - np.mean(la_seq)) / np.std(la_seq)

            T_total = len(coords)
            if T_total < pre_turns + 5:
                continue
            pseudo_peak = rng.randint(pre_turns, T_total)

            pre_mask = [(t >= pseudo_peak - pre_turns) and (t < pseudo_peak)
                        for t in la_turn_start]
            base_mask = [t < baseline_turns for t in la_turn_start]
            if sum(pre_mask) == 0 or sum(base_mask) == 0:
                continue

            pre_levy = np.mean(la_seq[pre_mask])
            baseline_levy = np.mean(la_seq[base_mask])
            pairs.append((baseline_levy, pre_levy))

        if len(pairs) < 5:
            continue
        T = np.mean([np.sign(b) == np.sign(p) for b, p in pairs])
        pseudo_Ts.append(T)
        if trial % 200 == 0:
            print(f"  trial {trial}/{n_trials} ({time.time() - t1:.1f}s)")

    pseudo_Ts = np.array(pseudo_Ts)
    n_exceed = int(np.sum(pseudo_Ts >= real_T))
    p_val_mc = (n_exceed + 1) / (len(pseudo_Ts) + 1)
    print(f"\nPseudo (n={len(pseudo_Ts)}): mean={pseudo_Ts.mean():.4f} "
          f"\u00b1 {pseudo_Ts.std():.4f}")
    print(f"Monte-Carlo p-value: {p_val_mc:.4f}")
    print(f"\u0394T = {(real_T - pseudo_Ts.mean()) * 100:+.1f} percentage points")
    return real_T, pseudo_Ts, p_val_mc


# ============================================================
# 2c. Matched pseudo-anchor test -- generic (ESConv, ANNO-MI)
#
# *** RECONSTRUCTED, NOT VERBATIM ***
# The exact original `matched_pseudo_anchor_test_generic` used for
# ESConv/ANNO-MI in earlier sessions could not be recovered. This
# reimplementation generalizes `matched_pseudo_anchor_test_supreme`
# above (same algorithm: real anchor from `anchor_key`, pseudo
# anchors sampled uniformly per session, same-orientation rate
# under both) to take pre-embedded sessions directly rather than
# raw text, since ESConv/ANNO-MI sessions are much shorter and do
# not require the same batched-encoding treatment as Supreme Court.
# Verify against the paper's reported values before treating this
# as authoritative:
#   ESConv:  N=27, T=.482, pseudo=.386+/-.087, p=.113
#   ANNO-MI: N=14, T=.357, pseudo=.402+/-.128, p=.683
# ============================================================

def matched_pseudo_anchor_test_generic(sessions, anchor_key='anchor',
                                        embeddings_key='embeddings',
                                        w=10, s=5, pre_turns=20,
                                        baseline_turns=15, n_trials=1000,
                                        corpus_name="", random_state=42):
    """
    sessions: list of dicts, each with pre-computed `embeddings_key`
    (a [T, D] array of per-turn embeddings) and a scalar
    `anchor_key` (the turn index of the true anchor).
    """
    rng = np.random.RandomState(random_state)

    coords_cache, anchors = {}, {}
    for i, sess in enumerate(sessions):
        emb = np.array(sess[embeddings_key])
        if len(emb) < w:
            continue
        pca = PCA(n_components=2)
        coords_cache[i] = pca.fit_transform(emb)
        anchors[i] = sess[anchor_key]

    def _levy_sequence(coords):
        la_seq, la_turn_start = [], []
        for start in range(0, len(coords) - w + 1, s):
            la_seq.append(compute_levy_area_2d(coords[start:start + w]))
            la_turn_start.append(start)
        if len(la_seq) < 4:
            return None, None
        la_seq = np.array(la_seq)
        if np.std(la_seq) > 1e-10:
            la_seq = (la_seq - np.mean(la_seq)) / np.std(la_seq)
        return la_seq, la_turn_start

    real_pairs = []
    for i, coords in coords_cache.items():
        la_seq, la_turn_start = _levy_sequence(coords)
        if la_seq is None:
            continue
        anchor = anchors[i]
        if anchor < pre_turns:
            continue
        pre_mask = [(t >= anchor - pre_turns) and (t < anchor) for t in la_turn_start]
        base_mask = [t < baseline_turns for t in la_turn_start]
        if sum(pre_mask) == 0 or sum(base_mask) == 0:
            continue
        pre_levy = np.mean(la_seq[pre_mask])
        baseline_levy = np.mean(la_seq[base_mask])
        real_pairs.append((baseline_levy, pre_levy))

    real_T = np.mean([np.sign(b) == np.sign(p) for b, p in real_pairs])
    print(f"[{corpus_name}] 실제 anchor 기준 T = {real_T:.4f} (N={len(real_pairs)})")

    pseudo_Ts = []
    for trial in range(n_trials):
        pairs = []
        for i, coords in coords_cache.items():
            la_seq, la_turn_start = _levy_sequence(coords)
            if la_seq is None:
                continue
            T_total = len(coords)
            if T_total < pre_turns + 5:
                continue
            pseudo_peak = rng.randint(pre_turns, T_total)
            pre_mask = [(t >= pseudo_peak - pre_turns) and (t < pseudo_peak)
                        for t in la_turn_start]
            base_mask = [t < baseline_turns for t in la_turn_start]
            if sum(pre_mask) == 0 or sum(base_mask) == 0:
                continue
            pre_levy = np.mean(la_seq[pre_mask])
            baseline_levy = np.mean(la_seq[base_mask])
            pairs.append((baseline_levy, pre_levy))
        if len(pairs) < 5:
            continue
        T = np.mean([np.sign(b) == np.sign(p) for b, p in pairs])
        pseudo_Ts.append(T)

    pseudo_Ts = np.array(pseudo_Ts)
    p_val = (int(np.sum(pseudo_Ts >= real_T)) + 1) / (len(pseudo_Ts) + 1)
    print(f"Pseudo-anchor (n={len(pseudo_Ts)}): "
          f"mean={pseudo_Ts.mean():.4f} \u00b1 {pseudo_Ts.std():.4f}")
    print(f"Monte-Carlo p-value: {p_val:.4f}")
    return real_T, pseudo_Ts, p_val


# ============================================================
# Reference values from the paper (for sanity-checking reruns)
# ============================================================

EXPECTED_HERO_TABLE = {
    # corpus: (N, T_true, pseudo_mean, pseudo_std, p)
    'Korean':  (71,    .704, .458, .059, .0000),
    'CMV':     (224,   .647, .421, .039, .0000),
    'ESConv':  (27,    .482, .386, .087, .1129),
    'ANNO-MI': (14,    .357, .402, .128, .6833),
    'Supreme': (1998,  .483, .452, .011, .0020),
}

EXPECTED_PAIRING_TABLE = {
    # corpus: (N, T, null_mean, null_std, p)
    'Korean':  (71,    .704, .497, .058, .0000),
    'CMV':     (1759,  .647, .473, .012, .0001),
    'ESConv':  (47,    .574, .484, .069, .158),
    'ANNO-MI': (31,    .548, .502, .090, .438),
    'Supreme': (500,   .522, .499, .022, .162),
}


if __name__ == '__main__':
    print("This module is meant to be imported. See docstring for the "
          "function corresponding to each corpus, and EXPECTED_HERO_TABLE / "
          "EXPECTED_PAIRING_TABLE above for the values reported in the paper.")
