# Benchmark: first-pass retrosynthesis settings on 80 random potent ChEMBL AChE actives and 10 known drugs, so the candidates' route-found rate has a reference. One process (RAM).
Set-Location "<PROJECT_ROOT>"
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'
& "C:\retro\venv\Scripts\python.exe" -u src\ad_v2_retro.py --in results\AD_V2_RETRO_BENCH_TODO.csv --out results\AD_V2_RETRO_BENCH.csv *> logs\ad_v2_retro_bench.log
