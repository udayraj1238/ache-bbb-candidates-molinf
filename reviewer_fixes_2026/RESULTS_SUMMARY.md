# Every result of the reviewer-weakness work, in one place

This is the full results log, kept as written during the work. **Read the section 'FINAL UPDATE 2026-11-05' at the end first**: it overrides the extended-set, docking and deep-retrosynthesis numbers given earlier in the file. File names refer to `reports/` (text reports), `candidate_tables/` (tables that contain structures) and `preregistrations/`.

# All results - FINAL snapshot 2026-11-02 14:10 IST (queue ended 13:45; supersedes numbers in the 2026-11-01 sections below where they differ)

## UPDATES since the 2026-11-01 snapshot (these override the older numbers further down)
- EXTENDED SET (AD_V4_FINAL_SUMMARY.txt, AD_V4_EXTENDED_SET.csv): 211 molecules, 96 scaffolds, 64 parents (65 first study + 146 full pool); 109 with scaffold absent from ChEMBL AChE; margin >= 10 %: 97; route <= 2 reactions: 129. Processed extras 651, of which 315 with a computed route; R1-flagged 274 of 651; carbamate series 162. Beating donepezil on predicted logBB (>= 0.75), CNS MPO (>= 4.1) and AChE p (>= 0.85): 5 (was 4).
- ROUND 3 (donepezil-oriented design pass, parents = extended-set molecules with predicted logBB >= 0.6; results/ext/extras3_*.csv): 97 new molecules -> novelty (first attempt failed on a PubChem outage, re-run: 85 checked, 85 clear, 0 unresolved) -> docking gate 35 pass -> patent/similarity search done (AD_V4_SIMILARITY_SEARCH_EXTRAS3.csv) -> local ring-breaker retrosynthesis (1000 it / 300 s): 19 of 35 solved (results/ext/deep_local_retro2.csv). These feed the 211-molecule extended set.
- ROUND 4 (second design pass from 25 parents with predicted logBB >= 0.45; results/ext/item2b_gated_gated.csv): 462 audited-gate survivors, 333 at margin >= 5 %. Downstream NOT completed: the novelty step could not run because the ChEMBL web service was down (HTTP 500 / timeouts, all 215 molecules unresolved); the queue correctly held the dependent steps. extras4_in.csv (247) and extras4_novelty.csv (215, unresolved) are on disk; rerun from the novelty step to finish.
- DEEP LOCAL RETROSYNTHESIS of the remaining unsolved extras (3000 it / 900 s, results/ext/deep_local_todo3.csv): NOT run (lane B was waiting on the round-4 novelty step until the end time).
- REINVENT 4 Mol2Mol BASELINE (item 23; AD_V4_MOL2MOL_BASELINE.txt, kaggle_reinvent/out): 9,936 sampled / 9,679 unique new from the first 450 filtered parents only (25 per parent; the run covered one chunk, 25 parents survive to the gates). Design-stage filters: 1,434 pass CNS MPO >= 4 and SA <= 3.5; 367 pass both BBB oracles; 345 pass AChE p >= 0.5. Audited gates 1-9: 55 survivors (28 at margin >= 5 %; 25 parents, 26 scaffolds). Per 1,000 design-stage survivors: 159, against 190 (BBB-mined edit library), 234 (AChE-informed library) and 135 (random edits). So the published generator performs between random edits and the mined library, on a small sample of parents.
- DOCKING scorers (AD_V4_RESCORE_REPORT.txt): Vinardo 0.686 (0.631-0.738; 678/680), Vina exhaustiveness 32 0.616 (0.549-0.673; 622/680), Vinardo exhaustiveness 32 0.588 (0.386-0.785; only 91/680 ligands scored, incomplete). Bar 0.70 not reached by any scorer (8 scorers tried after the first result).
- hERG: five feature sets, NOT MET (see section 5). ChEMBL logBB external test, patent search, calibration: unchanged from the sections below.
- Queue notes (logs/queue_A.log, logs/queue_B.log, logs/finish_auto.log, logs/finish2.log): lid-closed sleeps froze the queue several times (about 1 h each); the memory guard ended jobs when free commit fell under 0.35 GB; two silent failures were found and fixed (round-3 novelty passed with all sources unavailable; exit codes unreadable). Not completed: round-4 downstream, deep retro pass, Vinardo exhaustiveness 32.

# Earlier snapshot 2026-11-01 ~01:45 IST (kept for reference) - AChE / BBB candidate paper, reviewer-weakness fixes

Paper text NOT changed. Every number below comes from a file in `results/` (named per line). Negative results are kept. Pre-registration files: `results/PREREG_*.txt` (SHA-256 hashes in the files).
Running at the time of this snapshot: queue lanes A and B (`queue/`, `logs/queue_*.log`), Kaggle REINVENT Mol2Mol baseline (kernel `ache-reinvent`, v7).

## 1. Activity model (item 7)  - AD_V4_ACTIVITY_EVAL.txt, PREREG_ACTIVITY_V4.txt
- Delivered model kept (no candidate beat it). Final time-split tests (cutoffs 2020/2021/2022, AD frozen at 0.417): in-domain ROC-AUC 0.77 / 0.79 / 0.83; lower bounds above similarity and descriptor baselines at all three (E1, E2 met). Out of domain 0.71 / 0.76 / 0.80 (informational).
- Calibration (AD_V4_CALIBRATION.txt): ECE 0.03-0.07, but in-domain base rate of "active" is 0.83-0.89; precision at p >= 0.70 is 0.88 / 0.94 / 0.93 (about 5 points above base rate).
- BChE second target (AD_V4_ACTIVITY_EVAL_BCHE.txt): E1 met at 1 of 3 cutoffs, E2 at 3 of 3. BChE design: 90 audited-gate survivors from 4,000 seeds (84 at margin >= 5 %, 20 parents, 28 scaffolds).

## 2. Edit libraries / baselines (items 5, 23, 3) - AD_V4_ARMS.txt, AD_V4_GAN_BASELINE.txt
- Same generator, same gates: BBB-mined library 190 audited survivors per 1,000 design-stage survivors vs 135 for random edits (p 4e-8); AChE-preserving library (68 edits) 234 (p 8e-23).
- Project GAN (AChE+BBB fine-tuned, 30,000 samples, 28,848 valid, 6,171 unique): 0 pass CNS MPO >= 4 and SA <= 3.5 (median SA 6.9 in a 400 sample; 22/400 reach MPO 4, 1/400 SA 3.5). Zero survivors.
- REINVENT 4 Mol2Mol baseline: kernel v7 running (25 samples per parent, parents filtered for supported tokens); result file will be results/AD_V4_MOL2MOL_BASELINE.txt.

## 3. Full pool / extended set (items 14, 2, 3) - AD_V4_POOL_FUNNEL.txt, AD_V4_FINAL_SUMMARY.txt, AD_V4_EXTENDED_SET.csv
- No scaffold cap: design-stage survivors 5,832 (first study 2,700); audited gates 896 (374); margin >= 5 %: 653 (292); >= 10 %: 349 (159). AChE-informed library: 5,362 -> 1,103 -> 767 (>= 5 %) -> 410 (>= 10 %).
- Extras downstream (stereo, novelty, docking): 566 processed, 296 with a computed route (first Kaggle passes + local deep ring-breaker pass 1000 it/300 s solved 41 of 138). R1-flagged (known compound >= 0.80): 272; carbamate series 162.
- EXTENDED SET (route, not carbamate series, not R1-flagged): 193 molecules, 92 scaffolds, 59 parents (65 first study + 128 full pool); margin >= 10 %: 89; route <= 2 reactions: 123.
- Donepezil comparison (item 2): predicted logBB >= 0.75, CNS MPO >= 4.1, AChE p >= 0.85: 4 in the extended set (11 of the 15 in the pool are eseroline carbamates, excluded by the pre-registered rule). Donepezil-oriented design pass (parents: 18 extended-set molecules with logBB >= 0.6; results/ext/item2_gated_gated.csv): 187 survivors, 128 at margin >= 5 %; 9 non-carbamate designs (6 scaffolds) beat donepezil on all three properties, 5 of them new. Round 3 downstream (97 new molecules) is running in the queue.

## 4. BBB model independence / external tests (items 11, 16) - AD_V4_BBB_INDEPENDENCE.txt, AD_V4_EXTERNAL_BBB.txt, AD_V4_EXTERNAL_LOGBB.txt
- 94.7 % of Martins compounds are also in B3DB: the "independent" model is not independent.
- External ATC-labelled drugs (n = 488): AUC in-loop 0.83, audit 0.88, Martins 0.82, -TPSA 0.69.
- ChEMBL measured logBB, compounds in none of the training sets (n = 84; 77 BBB+, 7 BBB-): AUC 0.95 / 0.89 / 0.89; -TPSA 0.99; logBB regression Spearman 0.70. Only 7 negatives, intervals wide; polarity alone is as good.

## 5. Efflux, logBB, hERG (items 8, 9, 10) - AD_V4_EXPOSURE.txt, AD_V4_HERG_SUMMARY_*.txt
- Efflux: first-study inhibitor term AUC 0.56; only polarity measures pass (N+O count 0.65, TPSA 0.67); legacy P-gp/Kp,uu gate dropped.
- logBB: count-ECFP + descriptors rho 0.724 vs 0.716, RMSE 0.541 vs 0.543 (no practical gain).
- hERG, five feature sets, none passes "cisapride predicted above donepezil": view A (cisapride 5.37 vs donepezil 5.69), cnt_desc, comb, ChemBERTa-77M (CV rho 0.59, panel AUC 0.93, 5.70 vs 5.90), ChemBERTa + view A (CV rho 0.66, panel AUC 1.00, 5.52 vs 5.85). VERDICT NOT MET for all; hERG reported as a flag only.

## 6. Docking (items 6, 12) - AD_V4_ENS_DOCK_REPORT.txt, AD_V4_GNINA_REPORT.txt, AD_V4_RESCORE_REPORT.txt
- Bar of the first study: ROC-AUC >= 0.70 (property-only 0.513). All scorers fall short.
- Benchmark AUC (138 actives, 542 decoys): Vina 4EY7 0.639; 4EY4 0.635, 4EY5 0.665, 4EY6 0.611, 4M0E 0.653; ensemble best score 0.633, mean 0.642. GNINA affinity 0.639, CNN score 0.608, CNN affinity 0.569, consensus Vina + CNN score + CNN affinity 0.661. Vinardo 0.686 (0.632-0.732; 678/680 ligands). Vina exhaustiveness 32: 0.620 (617/680 ligands). Seven scorers were tried after the first result.
- Delivered set (106): gate passed in all 5 receptors by 83, in at least half by 101, in none by 0; between-receptor Spearman 0.74; GNINA CNN score >= 0.5 for 100 %.
- Running: Vinardo at exhaustiveness 32 (queue lane A).

## 7. MD v3 (item 13) - AD_V4_MD_V3.txt, PREREG_MD_V3.txt
- Donepezil control: M1 2.34 / 1.54 / 1.48 / 1.73 A over four seeds -> V1 NOT met (seed 11 > 2.0 A), V2 met (SD 0.39 A).
- Primary endpoint (14 potent vs 15 property-matched inactives): M1 AUC 0.51, M2 0.59, M3 0.59 (lower bounds 0.29-0.39), none significant. Series-inactive secondary: AUC 0.46-0.57, none met. Verdict: not discriminating; MD not used in any claim.

## 8. Routes (items 19, 20) - AD_V4_ROUTE_SECOND.txt, AD_V4_ROUTE_TOTALS.txt
- Route totals: 93 of 106 delivered routes have total reactions = linear steps, 13 have one more.
- Second planner (USPTO + ring-breaker, Retro*): 83 of 106 solved; same length 70, fewer 3, more 10; both planners solve and agree within one reaction: 82 (77 %). Deep local pass on extras solved 41 of 138.

## 9. Prior art (items 17, 21) - AD_V4_SIMILARITY_SEARCH_V2.csv, AD_V4_PATENT_SEARCH.csv, PREREG_NOVELTY_RULE.txt
- Nearest known ChEMBL/PubChem compound of the 106: median 0.77; 41 at ECFP4 >= 0.80; 1 at >= 0.90. Rule R1 (>= 0.80 flagged; carbamate series reported separately) pre-registered.
- Patent-linked PubChem neighbours of the 193 extended-set molecules: 70 have one at ECFP4 >= 0.70, 9 at >= 0.80 (all full pool; first-study molecules 0 at >= 0.80); median best patent-linked similarity 0.71. Structure search of public patent-derived records only.

## 10. Thresholds (item 15) - AD_V4_SENSITIVITY.txt
- Stored 2,700 designs: baseline 374 (292 at margin >= 5 %); 10 % stricter -> 20 (5 %); 20 % stricter -> 0; 10 % looser 690 / 351; 20 % looser 1,063 / 366. Delivered 106: margin >= 10 %: 63, >= 15 %: 26, >= 20 %: 2.

## 11. Pre-specified bars (item 18) - AD_V4_PRESPEC_BARS.txt (table of every bar and outcome)

## Not fixable by computation
1 wet-lab assays, 22 IP/patent opinion, 19 chemist review, writing items 4 and 25-29 (not started), wording of 6, 9, 10, 13.

## NOTE 2026-11-03 01:45
Queue extended to 2026-11-04 23:00 (lanes A and B, 4 CPU threads each at idle priority, tree-wide). The REINVENT 4 Mol2Mol baseline was only partly downloaded earlier: all 5 chunks exist on Kaggle (about 50,000 samples from all 2,216 parents); the full set is now being converted, filtered and gated in the queue (jobs M1-M4), and a replicate run (kernel version 8) is queued for a second random draw. Round-4 docking had stalled for 7 h (restart fixed it); deep retrosynthesis is slow (about 24 min per molecule on average, one took 3.2 h).

## FULL REINVENT 4 Mol2Mol BASELINE (2026-11-03; supersedes the partial 9,679-sample figure above)
All 5 chunks (56,313 sampled, 45,149 unique new) from all filtered parents: CNS MPO >= 4 and SA <= 3.5: 8,381; both BBB oracles >= 0.75: 2,565; AChE p >= 0.5: 2,464 design-stage survivors. Audited gates 1-9: 536 survivors (370 at validated margin >= 5 %, 212 at >= 10 %; 194 parents, 344 scaffolds; 47 in the carbamate series; 3 beat donepezil on predicted logBB/MPO/AChE p; 12 already in the extended set). Per 1,000 design-stage survivors: 217, against 190 (BBB-mined edit library), 234 (AChE-informed library) and 135 (random edits). Not taken through stereo / novelty / docking / routes. Files: results/AD_V4_MOL2MOL_BASELINE.txt, results/ext/gated_mol2mol_gated.csv.

## FINAL UPDATE 2026-11-05 (queue ended 2026-11-04 23:00; supersedes the extended-set and docking numbers above)
- EXTENDED SET (AD_V4_FINAL_SUMMARY.txt, AD_V4_EXTENDED_SET.csv): 403 molecules, 125 scaffolds, 92 parents (65 first study + 338 full pool); 207 with a scaffold absent from ChEMBL AChE; validated margin >= 10 %: 222; route <= 2 reactions: 269. Processed extras 866 (novelty-clear and docking gates passed), 536 with a computed route; R1-flagged 305; carbamate series 162. Beating donepezil on predicted logBB >= 0.75, CNS MPO >= 4.1 and AChE p >= 0.85: 8. Diverse subset: 60 molecules, 45 scaffolds, 51 parents.
- ROUND 4 (donepezil-oriented design pass 2, 25 parents with predicted logBB >= 0.45): 462 audited-gate survivors (333 at margin >= 5 %); novelty 212 of 215 clear (ChEMBL back); docking 211 pass; local ring-breaker retrosynthesis (1000 it / 300 s): 169 of 211 solved (results/ext/deep_local_retro4.csv).
- DEEP RETROSYNTHESIS (3000 it / 900 s, USPTO + ring-breaker) of the 162 extras that no earlier pass solved: all 162 processed, 52 solved (results/ext/deep_local_retro3_s0/s1/s2.csv).
- REINVENT 4 Mol2Mol: full baseline 536 survivors (370 at margin >= 5 %); replicate draw 491 survivors (349 at >= 5 %; 172 parents, 335 scaffolds): stable across draws (AD_V4_MOL2MOL_BASELINE.txt, ..._rep2.txt).
- DOCKING (AD_V4_RESCORE_REPORT.txt): Vinardo 0.686 (0.631-0.738; 678/680), Vina exhaustiveness 32 0.616 (0.549-0.673; 622/680), Vinardo exhaustiveness 32 0.673 (0.600-0.743; 235/680 ligands scored, incomplete). Bar 0.70 not reached by any scorer (8 scorers tried after the first result).
- Not changed: MD v3 (not discriminating, control V1 failed), hERG (five feature sets, NOT MET), GAN baseline (0 usable molecules), calibration caveat, BBB independence results, ChEMBL logBB external test, patent search (193 molecules; the molecules added later are covered by the similarity searches only).
- Open / not done: wet-lab assays, patent opinion, chemist review of routes, all paper-writing items; the 235/680 Vinardo e32 benchmark coverage; the patent-linked search was not rerun on the molecules added after the first 193.
