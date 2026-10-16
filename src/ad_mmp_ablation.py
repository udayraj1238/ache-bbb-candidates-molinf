"""
Does the matched-molecular-pair engine add anything beyond "make it less polar"?

WHY THIS EXISTS
Candidates are made by applying transformations from a library of 1,090 (mined from BBB
matched pairs, effect = change in out-of-fold PREDICTED BBB probability, with measured label
flips as a sign check). Many of the highest-effect transformations remove hydrogen-bond donors
or polarity, which any medicinal chemist would try anyway. This tests, on the independent
audit model, whether the library's statistics do better than (a) random transformations and
(b) a one-line polarity rule with no matched-pair statistics at all.

DESIGN, FIXED BEFORE RUNNING
  parents    300 random potent ChEMBL AChE actives (pChEMBL >= 6.5), as in the design run
  arms       LIB     the transformations the design used: mean_dP > 0 (confirmatory,
                     supporting or exploratory tier)
             RANDOM  transformations drawn uniformly from all 1,090 (either sign)
             TPSA    the same NUMBER of transformations as LIB, chosen only by the most negative
                     mean change in TPSA (no matched-pair statistic used)
  products   all products of each arm on the same parents, deduplicated
  read-out   change in the AUDIT BBB probability (product - parent). The audit model is not the
             model that produced the library statistics (an out-of-fold in-loop-type forest).
             Also: share of products with audit >= 0.75, and share with product AChE probability
             >= 0.70 (activity retained).
  pass bar   LIB shows a larger mean audit change than RANDOM (Mann-Whitney p < 0.01) AND a
             larger mean than TPSA with non-overlapping 95 % bootstrap intervals. If LIB is not
             better than TPSA, the matched-pair statistics add nothing beyond polarity reduction.

Run:  python src/ad_mmp_ablation.py      (bbbenv interpreter)
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from design_disease_candidates import apply_transform_indexed, fragment_index, canon  # noqa: E402,F401
from oracle_gap import load_oracles, score as bbb_score  # noqa: E402
from ad_hard_gates import proba  # noqa: E402

LIB = "results/mmp_transform_library_v2.csv"
ACHE = "data/processed/AChE_Alzheimers_curated.csv"
OUT = "results/AD_MMP_ABLATION.csv"
rng = np.random.default_rng(13)
N_PARENTS = 300


def products(idx, lib):
    seen, rows = set(), []
    for _, t in lib.iterrows():
        for prod, parent in apply_transform_indexed(idx, canon(t.frag_from), t.frag_to):
            key = (prod, parent)
            if key not in seen:
                seen.add(key)
                rows.append((prod, parent, t.transform))
    return pd.DataFrame(rows, columns=["product", "parent", "transform"])


def boot_mean_ci(x, n=2000):
    x = np.asarray(x)
    m = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])


def main():
    lib = pd.read_csv(LIB)
    print("library: %d transformations; Spearman(mean_dP, -mean_dTPSA) = %.2f; Spearman(mean_dP, -mean_dHBD) = %.2f"
          % (len(lib), spearmanr(lib.mean_dP, -lib.mean_dTPSA).statistic, spearmanr(lib.mean_dP, -lib.mean_dHBD).statistic),
          flush=True)
    pos = lib[lib.mean_dP > 0]
    print("  positive-effect transformations: %d; of these %.0f %% lower TPSA and %.0f %% remove an H-bond donor"
          % (len(pos), 100 * (pos.mean_dTPSA < 0).mean(), 100 * (pos.mean_dHBD < 0).mean()), flush=True)

    a = pd.read_csv(ACHE)
    par = a[(a.label == 1) & (a.pchembl_mean >= 6.5)].smiles_std.dropna().unique()
    par = list(rng.choice(par, N_PARENTS, replace=False))
    idx = fragment_index(par)
    print("parents: %d; fragment keys: %d" % (len(par), len(idx)), flush=True)

    k = len(pos)
    arms = {"LIB (mean_dP > 0)": pos,
            "RANDOM (any sign)": lib.sample(k, random_state=1),
            "TPSA rule (most negative dTPSA)": lib.nsmallest(k, "mean_dTPSA")}
    o_in, o_au = load_oracles()
    parent_au = dict(zip(par, bbb_score(o_au, par)))
    parent_ache = dict(zip(par, proba("disease_AChE_Alzheimers", par)))
    rows = []
    for name, sub in arms.items():
        pr = products(idx, sub)
        if pr.empty:
            print("  %s: no products" % name)
            continue
        pr = pr.drop_duplicates("product").reset_index(drop=True)
        au = bbb_score(o_au, pr["product"].tolist())
        ac = np.asarray(proba("disease_AChE_Alzheimers", pr["product"].tolist()), float)
        pr["d_audit"] = au - pr.parent.map(parent_au).values
        pr["audit"] = au
        pr["ache"] = ac
        pr["arm"] = name
        pr = pr.dropna(subset=["d_audit"])
        rows.append(pr)
        lo, hi = boot_mean_ci(pr.d_audit)
        print("  %-34s %6d products | mean d(audit BBB) %+.3f [%+.3f, %+.3f] | audit >= 0.75: %.1f %% | "
              "AChE >= 0.70: %.1f %% | both: %.1f %%"
              % (name, len(pr), pr.d_audit.mean(), lo, hi, 100 * (pr.audit >= 0.75).mean(),
                 100 * (pr.ache >= 0.70).mean(), 100 * ((pr.audit >= 0.75) & (pr.ache >= 0.70)).mean()), flush=True)
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(OUT, index=False)
    L = res[res.arm.str.startswith("LIB")].d_audit
    R = res[res.arm.str.startswith("RANDOM")].d_audit
    T = res[res.arm.str.startswith("TPSA")].d_audit
    p_r = mannwhitneyu(L, R, alternative="greater").pvalue
    llo, lhi = boot_mean_ci(L)
    tlo, thi = boot_mean_ci(T)
    print("\nLIB vs RANDOM: Mann-Whitney p = %.2g" % p_r)
    print("LIB %+.3f [%+.3f,%+.3f]  vs  TPSA rule %+.3f [%+.3f,%+.3f]" % (L.mean(), llo, lhi, T.mean(), tlo, thi))
    ok = (p_r < 0.01) and (llo > thi)
    print("PRE-SPECIFIED BAR (LIB > RANDOM at p < 0.01 AND LIB interval above the TPSA-rule interval)  ->  %s"
          % ("PASS" if ok else "FAIL"))
    print("wrote ->", OUT)


if __name__ == "__main__":
    main()
