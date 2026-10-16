"""Plain-language meaning of the brain-exposure correlations: pairwise ordering accuracy on the 1,025 measured logBB values.
For every pair of compounds whose measured logBB differs by at least 0.5 log units (a factor of about 3 in brain-to-blood ratio),
the fraction of pairs that a score orders correctly. Exact (all pairs), no sampling. Descriptive; no pass bar.
Also: share of compounds with logBB >= 0 in the top and bottom thirds of two scores.
Output: results/AD_PAIRWISE_MEANING.csv"""
import numpy as np
import pandas as pd
r = pd.read_csv("results/AD_EXPOSURE_LOGBB_ROWS.csv")
o = pd.read_csv("results/AD_LOGBB_OOF.csv")
rows = []


def pairwise(y, s, thr=0.5):
    dy = y[:, None] - y[None, :]
    ds = s[:, None] - s[None, :]
    m = np.triu(np.abs(dy) >= thr, 1)
    return float(((np.sign(dy) == np.sign(ds))[m]).mean()), int(m.sum())


for name, y, s in [("exposure score (passive x (1 - P-gp))", r.logBB.values, r.kpuu_surrogate.values),
                   ("passive-permeability term alone", r.logBB.values, r.p_passive_permeable.values),
                   ("1 - P-gp term alone", r.logBB.values, r.one_minus_pgp.values),
                   ("-TPSA", r.logBB.values, -r.tpsa.values),
                   ("logBB ensemble (scaffold-grouped CV)", o.logBB.values, o["fingerprint+descriptor ensemble"].values)]:
    f, n = pairwise(y, s)
    rows.append(dict(item="pairs ordered correctly", score=name, value=round(f, 3), n_pairs=n))
    print("%-45s %.3f  (%d pairs)" % (name, f, n))
y = r.logBB.values
for name, s in [("exposure score", r.kpuu_surrogate.values), ("-TPSA (lowest polar surface area = top)", -r.tpsa.values)]:
    q = np.quantile(s, [1 / 3, 2 / 3])
    top, bot = s > q[1], s <= q[0]
    rows.append(dict(item="share with logBB >= 0, top third", score=name, value=round(float((y[top] >= 0).mean()), 3), n_pairs=int(top.sum())))
    rows.append(dict(item="share with logBB >= 0, bottom third", score=name, value=round(float((y[bot] >= 0).mean()), 3), n_pairs=int(bot.sum())))
    print(name, "top %.2f bottom %.2f" % ((y[top] >= 0).mean(), (y[bot] >= 0).mean()))
pd.DataFrame(rows).to_csv("results/AD_PAIRWISE_MEANING.csv", index=False)
