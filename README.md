Lévy-Area Trajectory Archetypes across Dialogue Domains
Code for the paper: "Not Where, But How: Lévy-Area Trajectory Archetypes across Human Dialogue" (Submitted to EACL ARR August 2026)

Overview
We introduce a pre-event Lévy-area framework that quantifies signed rotational dynamics in semantic embedding space, and apply it to five dialogue corpora spanning Korean and English across counseling, persuasion, and institutional discourse.

Files
File	Description
levy_area.py	Core Lévy-area computation
clustering.py	Archetype discovery & validation
experiments.py	Ablation & anchor specificity
fig_pipeline.py	Figure 1: Pipeline overview
fig_stability.py	Figure 4: Cluster stability
fig_cmv.py	Figure 3: CMV replication
Requirements
pip install -r requirements.txt
Data
Corpus	Source	License
Korean counseling	AI Hub #71806	Research only
ESConv	Liu et al. (2021)	Public
ANNO-MI	Wu et al. (2022)	Public
CMV	Tan et al. (2016)	Public
Supreme Court	ConvoKit	Public
Place data files as follows:

data/sessions_with_risk.jsonl
embeddings/real_embeddings.npz
Reproducing Results
python clustering.py
python experiments.py
python fig_stability.py
python fig_pipeline.py
python fig_cmv.py
Key Results
Corpus	N	k	Silhouette	p
Korean	70	4	.400	<.0001
ESConv	47	4	.396	—
ANNO-MI	57	2	.448	—
CMV	1,759	4	.343	<.0001
Supreme	1,507	2	.587	<.0001
Anchor specificity: Real anchor sil = 0.556 vs. Random = 0.349 +/- 0.025 (p < .0001)

Feature redundancy: delta-LA only sil = 0.493 (k=4), same archetypes preserved

Citation
@inproceedings{anon2026levyarea,
  title={Not Where, But How: Lévy-Area Trajectory Archetypes across Human Dialogue},
  author={Anonymous},
  booktitle={ARR August 2026},
  year={2026}
}
License
Code: MIT License Data: See individual corpus licenses above.
