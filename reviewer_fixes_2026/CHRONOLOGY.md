# Chronology of the reviewer-weakness work (2026-10-25 to 2026-11-05)

Compute: a shared Windows laptop (other projects running; memory commit limit was exhausted repeatedly) plus Kaggle (CPU and GPU kernels, AiZynthFinder, OpenMM, Vina, GNINA, REINVENT 4). All times IST.

| Date | What happened |
|---|---|
| before 25 Oct | 106-molecule study finished; manuscript shortened with a Supplementary Information; the 29 weaknesses listed with a fixability table |
| 25 Oct | work started on every fixable weakness; pre-registrations written and hashed (`preregistrations/`): activity model (v4), novelty rule R1, MD v3 (+ series-inactive addendum). Activity model retrained/compared on fresh cutoffs 2020-2022 (item 7); edit-library arms vs random edits (items 5, 23, 3); uncapped full pool designed and gated (item 14); BBB independence and external BBB tests (items 11, 16); efflux, logBB, hERG attempts (items 8-10); route totals (item 20); similarity search of the 106 (items 17, 21); BChE as second target (item 24); MD v3 first batch (item 13) |
| 25-26 Oct | repeated out-of-memory kills (shared laptop) -> memory gate (`src/memgate.py`), thread caps, retry chains; Kaggle kernels for MD, ensemble docking, retrosynthesis of the new extras and a second route planner (item 19) |
| 26 Oct | ensemble docking and second planner re-run after errors (time limit; a Retro* cost-class bug); GNINA CNN rescoring on a Kaggle GPU; deeper local ring-breaker retrosynthesis; full-pool round 2 (AChE-informed library) and its downstream steps |
| 27-28 Oct | Vinardo and Vina exhaustiveness-32 re-docking of the benchmark; calibration of the activity model; ChEMBL measured-logBB external test; patent-linked PubChem search of 193 extended-set molecules |
| 29-30 Oct | donepezil-oriented design pass (round 3) and downstream; ChemBERTa embeddings on Kaggle for hERG; REINVENT 4 Mol2Mol baseline built on Kaggle (install, prior, token filtering, chunking) |
| 1-2 Nov | unattended two-lane queue with restart commands, memory guard, CPU throttle, idle priority, CPU-thread cap; found and fixed: round-3 novelty "passed" with every source unavailable, unreadable exit codes, report crash on partial file |
| 3-4 Nov | full Mol2Mol baseline (all 5 chunks) and a replicate; round 4 design pass; round 4 docking (7-hour stall fixed by restart), novelty, retrosynthesis; deepest local retrosynthesis of the 162 unsolved extras; final-set rebuilds |
| 5 Nov | queue ended 4 Nov 23:00; final snapshot; results log finalised |
| 11 Nov | repository reorganised (this folder); everything stored and indexed |

Failures worth knowing (all fixed, none changed a conclusion): memory exhaustion killed several model jobs; a Kaggle chemprop/numpy conflict (replaced by the project GNN); Kaggle dataset uploads that silently did not finish; a quoting bug in a generated notebook; stale "COMPLETE" kernel status read as a finished run; a docking step with all four novelty sources unavailable counted as passed (rerun); queue exit-code reading; a 7-hour docking stall.
