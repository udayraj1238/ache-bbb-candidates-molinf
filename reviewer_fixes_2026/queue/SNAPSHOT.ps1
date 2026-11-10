# Stores every result, log, script and queue definition in results\snapshot_<date> and commits it to the local repo (C:\bbbrep\ache_repo; NOT pushed).  Safe to run any time, any number of times.
param([string]$Tag = (Get-Date -Format "yyyy-MM-dd"))
Set-Location -Path $PSScriptRoot
$snap = "results\snapshot_$Tag"; New-Item -ItemType Directory -Force -Path "$snap\ext","$snap\kaggle_outputs","$snap\src","$snap\queue","$snap\logs" | Out-Null
Copy-Item results\AD_V4_*,results\PREREG_*,results\ALL_RESULTS_*.md,results\OVERNIGHT_STATUS.md,results\SYNTHESIS_SET.csv $snap -Force -ErrorAction SilentlyContinue
Get-ChildItem results\ext -File | Where-Object { $_.Length -lt 20MB } | Copy-Item -Destination "$snap\ext" -Force
foreach ($d in 'kaggle_retro\out_extras','kaggle_retro\out_extras2','kaggle_retro\out_extras_deep','kaggle_retro\out_second','kaggle_ens\out_gnina','kaggle_ens\out2_4EY4','kaggle_ens\out2_4EY5','kaggle_ens\out2_4EY6','kaggle_ens\out2_4M0E','kaggle_ens\bench_partial','kaggle_md3\out_ka','kaggle_md3\out_kb','kaggle_md3\out_kc','kaggle_act\out','kaggle_reinvent\out','kaggle_reinvent\out_rep2','kaggle_embed\out') {
    if (Test-Path $d) { $dest = "$snap\kaggle_outputs\" + ($d -replace '\\','_'); New-Item -ItemType Directory -Force -Path $dest | Out-Null; Get-ChildItem $d -Recurse -File -Include *.csv,*.txt,*.log,*.json | Where-Object { $_.Length -lt 20MB } | Copy-Item -Destination $dest -Force }
}
Copy-Item src\ad_v4_*.py,src\memgate.py,src\memrun.py,src\design_disease_candidates.py,src\dock_target.py,src\ad_v2_retro.py "$snap\src" -Force
Copy-Item queue\jobs_*.ps1,QUEUE_*.ps1,FINISH*.ps1,SNAPSHOT.ps1,RUN_LONG_QUEUE.ps1,RUN_NIGHT_LONG.ps1,RUN_ITEMS.ps1,RUN_GAN_BASE.ps1 "$snap\queue" -Force -ErrorAction SilentlyContinue
Get-ChildItem logs -File | Where-Object { $_.Name -match '^(queue_|finish|night_long|long_queue|patent|item11)' -and $_.Length -lt 5MB } | Copy-Item -Destination "$snap\logs" -Force
$here = (Get-Location).Path; $R = "C:\bbbrep\ache_repo"
if (Test-Path $R) {
    New-Item -ItemType Directory -Force -Path "$R\results\snapshot_$Tag" | Out-Null; Copy-Item "$here\$snap\*" "$R\results\snapshot_$Tag\" -Recurse -Force
    Push-Location $R; git add -A 2>&1 | Out-Null; git -c user.name=udayraj1238 -c user.email=udayraj1238@users.noreply.github.com commit -m "Snapshot of all results, logs, scripts and queue definitions, $Tag" 2>&1 | Select-Object -Last 1; Pop-Location
}
$sz = Get-ChildItem $snap -Recurse -File | Measure-Object Length -Sum
Add-Content "logs\queue_control.log" ("[{0}] SNAPSHOT {1}: {2} files, {3:N1} MB" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Tag, $sz.Count, ($sz.Sum / 1MB))
Write-Host ("Snapshot {0}: {1} files, {2:N1} MB (local commit only, not pushed)" -f $Tag, $sz.Count, ($sz.Sum / 1MB))
