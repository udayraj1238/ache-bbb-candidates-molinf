# Negative and unwelcome results (kept on purpose)

None of these were deleted, softened or re-defined. Each stays in the record because it limits what the paper can claim.

1. **Docking never reached its bar (item 6).** Pre-specified ROC-AUC >= 0.70 on the property-matched benchmark (138 actives, 542 decoys; property-only model 0.513). Best of eight scorers: Vinardo 0.686 (0.631-0.738). Docking is therefore a pose filter, not a selection criterion. (`reports/AD_V4_RESCORE_REPORT.txt`, `AD_V4_GNINA_REPORT.txt`, `AD_V4_ENS_DOCK_REPORT.txt`)
2. **hERG cannot be ranked (item 10).** Five feature sets, all NOT MET on "cisapride predicted above donepezil" (e.g. 5.37 vs 5.69; ChemBERTa 5.70 vs 5.90). hERG is a flag only. (`reports/AD_V4_HERG_SUMMARY_*.txt`)
3. **The MD screen is not valid (item 13).** The donepezil control fails (seed 11: 2.34 A > 2.0 A) and nothing separates 14 potent from 15 matched inactives (AUC 0.51 / 0.59 / 0.59, p 0.40-0.91). MD is used in no claim. (`reports/AD_V4_MD_V3.txt`)
4. **logBB features gave no gain (item 9).** rho 0.724 vs 0.716.
5. **The efflux term does not work (item 8).** AUC 0.56; only polarity measures pass (0.65-0.67). The legacy P-gp / Kp,uu gate was dropped.
6. **The BBB models are not independent (items 11, 16).** 94.7 % of the Martins compounds are also in B3DB. On measured logBB, plain -TPSA (AUC 0.99) is as good as any model (0.89-0.95), with only 7 BBB-negative compounds.
7. **The activity-model gate adds little beyond similarity in domain.** Precision at p >= 0.70 is 0.88 / 0.94 / 0.93 against an in-domain base rate of 0.83-0.89 (`reports/AD_V4_CALIBRATION.txt`). The ranking AUC (0.77-0.83) is the better evidence.
8. **The pipeline is fragile to its thresholds (item 15).** 10 % stricter numeric thresholds keep 5 % of the stored designs; 20 % stricter keep none.
9. **41 of the first 106 molecules have a known compound at ECFP4 >= 0.80 (item 21).** Patent-linked neighbours exist for 70 of 193 extended-set molecules at >= 0.70.
10. **Only a few designs beat donepezil on predicted exposure (item 2).** 8 of 403; the first-pool winners were eseroline carbamates.
11. **The project's own GAN yields nothing usable (item 23).** 0 of 6,171 unique valid samples pass CNS MPO >= 4 and SA <= 3.5 (median SA 6.9). The published generator (REINVENT 4 Mol2Mol) performs like the mined libraries (217 vs 190 and 234 survivors per 1,000), so the edit pipeline is not better than a published generator on this measure.
12. **Routes are computed, not tested (item 19).** Many extras remain unsolved: 110 of 162 stayed unsolved after the deepest pass.
13. **Process failures that were found and fixed (not results, but worth knowing):** a novelty step once "passed" with all four sources unavailable (outage); the queue once could not read exit codes; a docking step stalled for 7 hours; stale Kaggle statuses once looked like completed runs. All are described in `CHRONOLOGY.md`.
