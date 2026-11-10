# Reviewer-weakness work, October-November 2026

This folder documents the computational work done to fix, quantify or honestly report the 29 weaknesses a reviewer could flag in the paper *Computational Design of 106 Novel, Brain-Penetrant Acetylcholinesterase Inhibitor Candidates with Computed Synthetic Routes*. **The paper text was not changed.** Everything is a computational prediction; no molecule was synthesised or assayed.

## Start here
| File | What it is |
|---|---|
| [`ITEMS_STATUS.md`](ITEMS_STATUS.md) | the 29 weaknesses, one row each: status, what was found, the evidence file |
| [`NEGATIVE_RESULTS.md`](NEGATIVE_RESULTS.md) | everything that did not work or limits the claims |
| [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md) | every result with its numbers (read the *FINAL UPDATE* section at the end first) |
| [`CHRONOLOGY.md`](CHRONOLOGY.md) | what was done when, and the process failures that were found and fixed |
| [`REPRODUCE.md`](REPRODUCE.md) | how to rerun any step, the Kaggle jobs and the unattended queue |

## Headline numbers (final, 2026-11-05)
* **Extended candidate set: 403 molecules** (125 scaffolds, 92 parents; the first study's 106 are included), each with a computed route, novelty checks and docking/pose gates; 207 have a scaffold absent from ChEMBL AChE; 8 beat donepezil on predicted logBB, CNS MPO and AChE probability together.
* **Activity model passed its pre-registered fresh-cutoff test** (in-domain ROC-AUC 0.77 / 0.79 / 0.83) - but precision at the gate is only about 5 points above the in-domain base rate.
* **Edit libraries beat random edits** (190 / 234 vs 135 audited survivors per 1,000), and a published generator (REINVENT 4 Mol2Mol) lands at 217 per 1,000, so the edit pipeline is not better than a published generator on this measure.
* **Not fixed (reported as limits):** docking enrichment (best ROC-AUC 0.686, bar 0.70, eight scorers), hERG ranking (five feature sets), MD (control fails), logBB (no gain), BBB-model independence (94.7 % overlap).

## Layout of this folder (as committed)
| Path | Content |
|---|---|
| `reports/` | the text reports behind every number (`AD_V4_*.txt`) |
| `preregistrations/` | the three hashed pre-registrations |
| `queue/` | the unattended queue (runner, job lists, stop/start/status/snapshot scripts, follow-up pollers) |
| `kaggle/` | notebook generators, kernel metadata and small dataset scripts of every Kaggle job (see `REPRODUCE.md`) |
| `logs/` | queue and follow-up logs (stored as `.log.txt`) |
| `candidate_tables/` | tables that contain molecule structures (gated designs, extended set, routes, baselines, similarity and patent searches). **Held back from the public repository until the authors decide on disclosure (patent considerations, item 22);** kept in the local project folder and in the private backup. |

New and changed scripts are in the repository's `src/` (`ad_v4_*.py`, `memgate.py`, `memrun.py`, plus edits to `design_disease_candidates.py`, `dock_target.py`, `ad_v2_retro.py`, `ad_v2_dock.py`).
