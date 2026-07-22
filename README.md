# Lévy-Area Trajectory Archetypes across Dialogue Domains

Code for the paper: "Not Where, But How: Lévy-Area Trajectory
Archetypes across Human Dialogue" (Submitted to EACL ARR August 2026)

## Overview

We introduce a pre-event Lévy-area framework that quantifies signed
rotational dynamics in semantic embedding space, and apply it to
five dialogue corpora spanning Korean and English across counseling,
persuasion, and institutional discourse.

## Repository Structure
├── src/
│   ├── levy_area.py      — Core Lévy-area computation
│   ├── clustering.py     — Archetype discovery & validation
│   └── experiments.py    — Ablation & anchor specificity
├── figures/
│   ├── fig1_archetypes.py          — Figure 1: Discovery archetypes (Korean, k=4)
│   ├── fig2_cmv_replication.py     — Figure 2: CMV cross-corpus replication
│   ├── fig3_stability.py           — Figure 3: Bootstrap / permutation / anchor specificity
│   ├── appendix_a1_cross_corpus.py — Appendix Figure A1: Korean–ESConv overlay
│   ├── appendix_a2_cluster_overlay.py — Appendix Figure A2: Cluster 1 vs 3 curves
│   ├── appendix_a3_clinical.py     — Appendix Figure A3: ANNO-MI clinical relevance
│   └── appendix_a4_dendrogram.py   — Appendix Figure A4: Hierarchical clustering structure
└── data/                 — (not included; see Data section below)

## Requirements
pip install numpy scipy scikit-learn matplotlib pandas

## Data

| Corpus | Source | License |
|---|---|---|
| Korean counseling | AI Hub #71806 | Research only |
| ESConv | Liu et al. (2021) | Public |
| ANNO-MI | Wu et al. (2022) | Public |
| CMV | Tan et al. (2016) | Public |
| Supreme Court | ConvoKit | Public |

**Note on data access**: The Korean counseling corpus requires
research registration via AI Hub (#71806) and cannot be
redistributed; it is not included in this repository. Public
corpora (ESConv, ANNO-MI, CMV, Supreme Court) are accessible
directly via their original sources listed above.

Place preprocessed files as follows before running `src/clustering.py`:
data/sessions_with_risk.jsonl
embeddings/real_embeddings.npz

`src/clustering.py` additionally expects intermediate pickled
objects (`pre_results_clean.pkl`, `bal_features.pkl`, `X3.npy`)
generated from the raw session data. Update the `BASE` path
variable at the top of the script to point to your local data
directory (the released code was run on Google Colab with a
mounted Drive path; this has been generalized to a local `data/`
path for portability).

## Reproducing Results

Scripts must be run in the following order, since figure scripts
depend on in-memory variables produced by `src/clustering.py`
(`X3_scaled`, `labels4_clean`, `km4`, `pre_results_clean`,
`bal_features`) rather than reloading them independently:

python src/clustering.py                       # generates core archetype variables
python src/experiments.py                      # ablation & anchor specificity (N=87 subset)
python figures/fig1_archetypes.py
python figures/fig2_cmv_replication.py
python figures/fig3_stability.py
python figures/appendix_a1_cross_corpus.py
python figures/appendix_a2_cluster_overlay.py
python figures/appendix_a3_clinical.py
python figures/appendix_a4_dendrogram.py


Steps 3–9 assume the variables from step 1 are available in the
same session/namespace (e.g., run within a single notebook session,
or refactor `src/clustering.py` into an importable module).

## Key Results

| Corpus | N | k | Silhouette | p |
|---|---|---|---|---|
| Korean | 70 | 4 | .400 | <.0001 |
| ESConv | 47 | 4 | .396 | — |
| ANNO-MI | 57 | 2 | .448 | — |
| CMV | 1,759 | 4 | .343 | <.0001 |
| Supreme | 1,507 | 2 | .587 | <.0001 |

- Anchor specificity: Real anchor sil = 0.556 vs. Random = 0.349 ± 0.025 (p < .0001)
- Feature redundancy: ΔLA only sil = 0.493 (k=4), same archetypes preserved

## Citation

```bibtex
@inproceedings{anon2026levyarea,
  title={Not Where, But How: Lévy-Area Trajectory Archetypes across Human Dialogue},
  author={Anonymous},
  booktitle={ARR August 2026},
  year={2026}
}
```

## License

Code: MIT License. Data: See individual corpus licenses above.
