# How to reproduce and extend the work

All paths are relative to the repository root. Environment: Python 3.13 (project venv; see `requirements.txt`), AutoDock Vina 1.2.7 at `docking/vina(.exe)`, AiZynthFinder 4.4.1 in a separate Python 3.12 venv with the public USPTO / ring-breaker models and the ZINC stock (see `src/ad_v2_retro.py`), network access to ChEMBL and PubChem. Kaggle steps need a Kaggle account and `kaggle` CLI.

## Pipeline order (scripts in `src/`)
1. **Gates for any design table:** `ad_v4_gates.py OUT_PREFIX IN.csv [--relax r] [--target ache|bche]` (audited gates 1-9 and the validated margin; reproduces 374 -> 292 at >= 5 %, 159 at >= 10 %).
2. **Design from parents:** `design_disease_candidates.py --disease alzheimers [--library LIB --library-mode all --max-per-scaffold N --parents-file F --round2-start S --round2-seeds E]`.
3. **Downstream of new molecules:** `ad_v4_extras.py collect|stereo|offline|novelty|dock|toretro` (set `EXTRAS_TAG` to the round: "" 1, "2", "3", "4"); `ad_v4_similarity_search.py IN OUT`; `ad_v4_patent_search.py`; retrosynthesis `ad_v2_retro.py` (first pass 200 it / 120 s; deeper passes 1000 it / 300 s and 3000 it / 900 s with `RETRO_EXP=uspto,ringbreaker`).
4. **Assemble the extended set and reports:** `ad_v4_final_set.py`, `ad_v4_pool_report.py`, `ad_v4_sensitivity_report.py`, `ad_v4_arms_report.py`, `ad_v4_rescore_report.py`, `ad_v4_ens_report.py`, `ad_v4_gnina_report.py`, `ad_v4_mol2mol_report.py`, `ad_v4_gan_report.py`, `ad_v4_calibration.py`.
5. **Model checks:** `ad_v4_activity_fit.py` / `ad_v4_activity_eval.py` (`ACT_TARGET=ache|bche`), `ad_v4_exposure.py`, `ad_v4_herg.py a|cnt_desc|comb|chem|chem_a` (`dump` writes the structures for the embedding kernel), `ad_v4_bbb_independence.py`, `ad_v4_external_bbb.py`, `ad_v4_external_logbb.py`, `ad_v4_md_analysis.py`, `ad_v4_rescore_local.py TAG EXH WORKERS` (`VINA_SCORING=vinardo` for Vinardo).
6. **Baselines:** `ad_v4_gan_baseline_sample.py` + `ad_v4_gan_baseline_prep.py` (also used for REINVENT output) + `ad_v4_gates.py`; REINVENT 4 Mol2Mol on Kaggle (`kaggle/reviewer_fixes_2026/kaggle_reinvent`), then `ad_v4_mol2mol_convert.py` -> prep -> gates -> report (`M2M_TAG=_rep2` for the replicate).

## Kaggle jobs (`kaggle/reviewer_fixes_2026/`)
`kaggle_act` (GNN + ChemBERTa activity models), `kaggle_ens` (ensemble docking, GNINA), `kaggle_retro` (retrosynthesis, second planner, deep pass), `kaggle_md3` (MD v3 and series set), `kaggle_embed` (ChemBERTa embeddings), `kaggle_reinvent` (REINVENT 4). Each folder has its dataset-building scripts, notebook generator and kernel metadata; large inputs are not committed. Upload datasets with `kaggle datasets create -p <folder> --dir-mode zip` from inside the parent folder (relative paths with a slash broke the CLI), then push the kernel.

## Unattended queue (`queue/`)
`QUEUE_RUNNER.ps1 -Lane A|B` runs the job lists `jobs_A.ps1` / `jobs_B.ps1` at idle priority on a fixed CPU-thread mask, suspends under CPU load, ends its job on memory shortage (progress is kept), and retries failures. Control: `QUEUE_STOP.ps1` (stop everything, no progress lost), `QUEUE_START.ps1` (resume), `QUEUE_STATUS.ps1`, `SNAPSHOT.ps1` (store results and commit locally). Every job is resumable from its own output files.

## Pre-registrations
`preregistrations/PREREG_ACTIVITY_V4.txt`, `PREREG_NOVELTY_RULE.txt`, `PREREG_MD_V3.txt` (with SHA-256 hashes). Bars were fixed before the runs and never moved.
