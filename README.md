# Not Where, But How: Lévy-Area Dynamics Reveal Transition-Specific Coupling in Dialogue

Code for the paper submitted to EACL ARR (August 2026 cycle).

## Overview

We introduce a Lévy-area framework for quantifying rotational
dynamics between a session's baseline and pre-transition windows.
Because Lévy area is computed from session-specific PCA
projections, absolute directional identity is not comparable
across sessions; we therefore define a **sign-invariant
orientation relation** (same-vs-different sign between baseline
and pre-transition rotation) as the primary unit of analysis.

Using this relation, we test:
1. **Pairing permutation** — does baseline--pre-event coupling
   exceed chance?
2. **Matched pseudo-anchor test** — is that coupling specific to
   the true transition, or a general session-wide property?

across five dialogue corpora spanning Korean and English:
Korean counseling, ESConv, ANNO-MI, CMV persuasion, and US
Supreme Court oral arguments.

Exploratory signed clustering is reported separately as
descriptive evidence only; cluster identity is unstable under
the sign ambiguity of session-wise PCA (Appendix).

## Repository Structure

```
├── src/
│   ├── levy_area.py            — Core Lévy-area computation
│   ├── preprocessing.py        — Corpus-specific session filtering
│   │                              (Korean risk-event sessions,
│   │                              Supreme Court advocate-only
│   │                              filtering)
│   ├── orientation_tests.py    — Pairing permutation and matched
│   │                              pseudo-anchor tests (primary
│   │                              confirmatory results)
│   └── exploratory_clustering.py — Exploratory signed clustering
│                                    (k selected by silhouette)
├── figure/
│   ├── fig1_concept.py         — Figure 1: Lévy-area geometry and
│   │                              the sign-invariant orientation
│   │                              relation
│   ├── fig2_hero_effect.py     — Figure 2: transition-specific
│   │                              enrichment across corpora
│   └── appendix_dendrogram.py  — Appendix Figure A1: exploratory
│                                   cluster structure (Korean
│                                   corpus)
└── data/  — (not included; see Data section below)
```

## Requirements

```
pip install numpy scipy scikit-learn matplotlib pandas sentence-transformers
```

## Data

| Corpus | Source | License |
|---|---|---|
| Korean counseling | AI Hub #71806 | Research only |
| ESConv | Liu et al. (2021) | Public |
| ANNO-MI | Wu et al. (2022) | Public |
| CMV | Tan et al. (2016) | Public |
| Supreme Court | ConvoKit | Public |

**Note on data access:** The Korean counseling corpus requires
research registration via AI Hub (#71806) and cannot be
redistributed; it is not included in this repository. Public
corpora are accessible directly via their original sources.

`src/preprocessing.py` expects raw session data and produces the
intermediate files consumed by the rest of the pipeline
(`bal_features.pkl`: per-session embeddings and risk-turn labels).
Update the `BASE` path variable at the top of each script to
point to your local data directory.

## Reproducing Results

```
python src/preprocessing.py           # builds bal_features.pkl for all corpora
python src/orientation_tests.py       # pairing permutation + matched pseudo-anchor
                                       # (Tables: pairing, hero)
python src/exploratory_clustering.py  # exploratory clustering, k selected by
                                       # silhouette (Appendix Tables B1, B2)
python figure/fig1_concept.py
python figure/fig2_hero_effect.py
python figure/appendix_dendrogram.py
```

Each script reloads its inputs from disk rather than relying on
in-memory variables from a prior script, so they can be run
independently once `bal_features.pkl` (and the corpus-specific
equivalents) exist.

## Key Results

**Primary (confirmatory): transition-specific orientation
coupling**

| Corpus | $N$ | $T_{\rm true}$ | $\Delta T$ (pp) | $p$ |
|---|---|---|---|---|
| Korean counseling | 71 | .704 | +24.6 | <.001 |
| CMV | 224 | .647 | +22.6 | <.001 |
| ESConv | 27 | .482 | +9.5 | .113 (n.s.) |
| ANNO-MI | 14 | .357 | -4.5 | .683 (n.s.) |
| Supreme Court | 1,998 | .483 | +3.0 | .002 |

$\Delta T$ = true-anchor same-orientation rate minus mean
matched-pseudo-anchor rate. See paper Table 1 / Appendix A for
full pseudo-anchor means and standard deviations.

**Secondary (exploratory): signed clustering**

Korean corpus ($N{=}71$): $k{=}3$ (silhouette-selected,
silhouette$=.465$). Cluster identity is unstable under random
PCA-sign reflection (ARI$=.09$ against the unreflected solution,
essentially chance level; Appendix). We therefore do not treat
this clustering as a validated cross-domain typology.

## Citation

```bibtex
@inproceedings{anon2026levyarea,
  title={Not Where, But How: Lévy-Area Dynamics Reveal Transition-Specific Coupling in Dialogue},
  author={Anonymous},
  booktitle={ARR August 2026},
  year={2026}
}
```

## License

Code: MIT License. Data: see individual corpus licenses above.
