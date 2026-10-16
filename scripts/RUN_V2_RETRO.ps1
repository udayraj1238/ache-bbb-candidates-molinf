# Computed retrosynthesis for the 217 molecules that passed the offline gates; 2 shards in parallel (stock is held in RAM per process). Low priority.
Set-Location "<PROJECT_ROOT>"
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'
$py = "C:\retro\venv\Scripts\python.exe"
$p1 = Start-Process $py -ArgumentList "-u","src\ad_v2_retro.py","--in","results\AD_V2_OFFLINE.csv","--only-pass","--out","results\AD_V2_RETRO_s0.csv","--shard","0","--nshards","2" -RedirectStandardOutput logs\ad_v2_retro_s0.log -RedirectStandardError logs\ad_v2_retro_s0.err -PassThru -WindowStyle Hidden
$p2 = Start-Process $py -ArgumentList "-u","src\ad_v2_retro.py","--in","results\AD_V2_OFFLINE.csv","--only-pass","--out","results\AD_V2_RETRO_s1.csv","--shard","1","--nshards","2" -RedirectStandardOutput logs\ad_v2_retro_s1.log -RedirectStandardError logs\ad_v2_retro_s1.err -PassThru -WindowStyle Hidden
foreach ($p in $p1,$p2) { $p.PriorityClass = 'BelowNormal' }
$p1.WaitForExit(); $p2.WaitForExit()
