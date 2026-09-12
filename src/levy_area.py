"""
src/levy_area.py

Core Lévy-area computation (paper §3.4, Eq. 1). This is the single
source of truth for this quantity -- src/preprocessing.py,
src/orientation_tests.py, and src/exploratory_clustering.py all
import from here rather than redefining it, to avoid the kind of
silent divergence that produced the A105 indexing bug (see
src/preprocessing.py module docstring).

*** Note on an earlier version of this file ***
An earlier version standardized (z-scored) the session's 2D PCA
coordinates BEFORE computing the per-window Lévy area. The
pipeline actually used for every result reported in the paper does
the opposite: it computes the raw Lévy area for each sliding
window first, and z-scores the resulting session-level *sequence*
of window-wise Lévy areas afterward (paper §3.4: "The resulting
sequence of window-level Lévy areas is standardized (z-scored)
within each session, prior to sign extraction and baseline/
pre-event averaging."). These are not interchangeable -- z-scoring
the coordinates changes the scale (and, for non-uniform turn
lengths, the shape) of each window's rotational signal before the
antisymmetric sum is taken. Use `sliding_window_levy_sequence`
below, not the coordinate-standardizing version, for anything that
needs to match the paper's reported numbers.
"""

import numpy as np
from sklearn.decomposition import PCA


def compute_levy_area_2d(coords):
    """
    Signed Lévy area (antisymmetric level-2 path-signature term,
    paper Eq. 1) of a 2D path.

    coords: array of shape (T, 2).
    """
    x, y = coords[:, 0], coords[:, 1]
    dx, dy = np.diff(x), np.diff(y)
    return 0.5 * np.sum(x[:-1] * dy - y[:-1] * dx)


def session_pca_coords(embeddings, n_components=2, random_state=None):
    """
    Fits PCA once on a session's full embedding sequence and
    returns the low-dimensional coordinates (paper §3.4: "For each
    session, PCA is fitted once using all eligible utterances,
    prior to extracting the baseline, pre-event, and pseudo-anchor
    windows.").
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    return pca.fit_transform(np.asarray(embeddings))


def sliding_window_levy_sequence(coords, w, s):
    """
    Computes the session-level sequence of window-wise Lévy areas
    (window width `w`, stride `s` between successive window start
    indices), then z-scores that sequence within-session.

    Returns (la_seq, la_turn_start) where la_seq is the z-scored
    Lévy-area sequence and la_turn_start gives the turn index at
    which each window begins (needed to select the baseline /
    pre-event / pseudo-anchor sub-windows downstream). Returns
    (None, None) if fewer than 4 windows can be formed.
    """
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


def window_mean(la_seq, la_turn_start, lo, hi=None):
    """
    Mean of la_seq over windows whose start index falls in
    [lo, hi) (or < lo if hi is None, i.e. a baseline-style window).
    Returns None if no windows fall in range.
    """
    if hi is None:
        mask = [t < lo for t in la_turn_start]
    else:
        mask = [(t >= lo) and (t < hi) for t in la_turn_start]
    if sum(mask) == 0:
        return None
    return np.mean(la_seq[mask])
