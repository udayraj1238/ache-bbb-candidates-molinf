"""Item 23: generative baseline.  Funnel of the project's own graph GAN (AChE + BBB fine-tuned, 30,000 samples) through the design-stage filters and the audited gates 1-9, set against the edit arms
(results/AD_V4_ARMS.txt).  Output: results/AD_V4_GAN_BASELINE.txt"""
import os
import pandas as pd
L = ["GAN baseline (results/ckpt_V9_ache_dual_s1.pt), 30,000 samples, same audited gates as the edit arms"]
for f in ("logs/gan_sample_1.out", "logs/gan_prep_1.out"):
    if os.path.exists(f):
        L += ["   " + l.strip() for l in open(f, errors="ignore") if l.strip() and not l.startswith("Traceback")][:8]
if os.path.exists("results/ext/gated_gan_funnel.csv"):
    F = pd.read_csv("results/ext/gated_gan_funnel.csv"); L += ["   gate %-66s %6d -> %6d" % (r.step, r.before, r.after) for r in F.itertuples()]
    G = pd.read_csv("results/ext/gated_gan_gated.csv"); L.append("   survivors of gates 1-9: %d | validated margin >= 5 %%: %d" % (len(G), int((G.min_slack_validated >= 0.05).sum()) if len(G) else 0))
else:
    L.append("   gates not reached (see logs/gan_gates_1.out)")
open("results/AD_V4_GAN_BASELINE.txt", "w").write("\n".join(L)); print("\n".join(L))
