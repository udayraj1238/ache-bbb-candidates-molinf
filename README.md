# Brain-penetrant acetylcholinesterase inhibitor candidates with computed synthetic routes

Code, data, results and manuscript for the study *Computational Design of 106 Novel, Brain-Penetrant Acetylcholinesterase Inhibitor Candidates with Computed Synthetic Routes* (U. Raj, M. Aggarwal, I. Sharma, C. M. Dasari), prepared for *Molecular Informatics*.

> **All results are computational predictions. No molecule has been synthesized or assayed.** The delivered structures are candidates for a first experimental test, not demonstrated inhibitors.

## What the study does

2,700 analogues of potent published AChE inhibitors, made by one or two matched-pair edits, were passed through a gated cascade (activity, blood-brain-barrier penetration by three independent models and a logBB regression, drug-likeness, toxicity relative to the parent, docking and pose coverage, protonation, tautomer and stereochemistry checks, a four-source novelty search against PubChem, ChEMBL and SureChEMBL, and computed retrosynthesis with AiZynthFinder). **106 molecules on 63 scaffolds** cleared every gate with at least 5 % margin (63 with 10 % or more); a **shortlist of 20** was ranked under 5,000 random weightings of eight further checks. Every model behind a gate was tested on held-out scaffolds against simple baselines, and on compounds published after a cutoff year; the components that did not hold up (docking score, the P-gp term, hERG, and the activity model for chemistry unlike its training set) are reported in the paper.

## Repository layout

| Folder | Content |
|---|---|
| `paper/` | `main.tex`, `References.bib`, `wileyref.bst`, figures, built `main.pdf`, Word manuscript and cover letter |
| `supporting_information/` | the 106 delivered structures (`SYNTHESIS_SET_106.csv`/`.sdf`) with every evidence column, the rejected candidates with the reason for each, the shortlist, computed routes, validation tables, the dossier |
| `src/` | every Python script of the study (design, gates, validation, docking, retrosynthesis, selection, figures); each script's header states its design and pass criterion |
| `scripts/` | the PowerShell drivers used to run the long jobs (Windows) |
| `data/` | curated training and validation tables (`data/processed`), publication years for the time-split test (`data/raw`) |
| `docking/` | PDB 4EY7, the prepared receptor and the docking box |
| `models/` | the trained scikit-learn models the gates use (`joblib`, scikit-learn 1.9.0) |
| `results/` | the result tables behind every number in the paper |
| `kaggle/` | the GPU jobs (explicit-solvent MD, pose stability, Boltz-2) as Kaggle notebooks with their input scripts |

## Setup

```bash
python -m venv venv && source venv/bin/activate      # Python 3.13 was used
pip install -r requirements.txt
```

* AutoDock Vina 1.2.7 (binary) must be placed at `docking/vina` (or `vina.exe`); it is not redistributed here. <https://github.com/ccsb-scripps/AutoDock-Vina/releases>
* Retrosynthesis used AiZynthFinder 4.4.1 in a separate Python 3.12 environment with its public USPTO models; see `src/ad_v2_retro.py` for the two search configurations.
* MD runs need OpenMM 8 with CUDA; see `kaggle/md_shortlist/`.
* Paths in `scripts/*.ps1` start with `<PROJECT_ROOT>` and `python`; replace them with your checkout and interpreter.

## Reproducing the paper's numbers

Run scripts from the repository root. The order used for the validation experiments and the figures:

1. `src/ad_ache_baselines.py`, `ad_bbb_baselines.py`, `ad_exposure_logbb_validation.py`, `ad_pgp_gate_validation.py`, `ad_random_vs_scaffold.py`, `ad_mmp_ablation.py`, `ad_split_deterministic_same_models.py`, `ad_paired_oracle_value.py`: held-out validation of the gate models (re-run from scratch, results reproduced; see `results/REAUDIT_COMPARISON.txt`).
2. `src/ad_redock_rmsd.py`, `ad_dock_enrichment_matched.py` then `ad_dock_enrichment_matched_summary.py`: docking validation.
3. `src/ad_fetch_chembl_years.py` then `ad_time_split_validation.py`: the time-split test.
4. `src/ad_v2_select.py --n 106 --per-scaffold 99 --per-parent 99 --exclude-carbamates`: the delivered set; `src/ad_v2_final_audit.py`: the 28-check independent audit.
5. `src/ad_v2_best.py`: the shortlist; `src/ad_v2_paper_facts.py`: every number quoted in the text.
6. `src/ad_v2_finalize_paper.py`: rebuilds figures, tables, the PDF and the Word file.

Note: the exposure checks are deterministic for fixed seeds; PubChem and SureChEMBL novelty queries depend on the databases' state on the query date (the cached answers are in `results/`).

## Data sources and licences

* **Our code** is under the MIT licence (`LICENSE`).
* **Our generated structures, tables and the manuscript text** are under CC BY 4.0 (`LICENSE-DATA`).
* Third-party data keep their own licences: ChEMBL (CC BY-SA 3.0), BindingDB, Therapeutics Data Commons and B3DB benchmarks, PDB entry 4EY7, ZINC, USPTO reaction data. Curated derivatives in `data/processed/` are provided for reproducibility and remain subject to the source licences.

## Citation

See `CITATION.cff`. A permanent DOI (Zenodo) will be added to the paper and this file at acceptance.

## Contact

Corresponding author: C. M. Dasari, Indian Institute of Information Technology Design and Manufacturing Kurnool (chandramohan.d@iiitk.ac.in).
