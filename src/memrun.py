"""python src/memrun.py GB script.py [args]: wait until GB of commit memory are free (src/memgate.py), then run the script in this process."""
import os, runpy, sys
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memgate
if os.path.exists("results/RETRO_ON_KAGGLE") and sys.argv[2].endswith("ad_v2_retro.py"):
    print("retrosynthesis of the extras runs on Kaggle CPU kernels (kaggle_retro/); nothing to do on the laptop"); sys.exit(0)
memgate.wait(float(sys.argv[1]))
script = sys.argv[2]; sys.argv = [script] + sys.argv[3:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name="__main__")
