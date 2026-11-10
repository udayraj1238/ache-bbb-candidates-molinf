# Stops the overnight queue completely and frees the CPU: pauses both lanes, disables their scheduled tasks and kills the running jobs.
# NO PROGRESS IS LOST: every job keeps its results in its own output files and resumes from them.  Restart with QUEUE_START.ps1.
Set-Location -Path $PSScriptRoot
New-Item -ItemType Directory -Force -Path "queue" | Out-Null
Set-Content "queue\PAUSE" (Get-Date -Format s)
foreach ($t in @("EXT_QA","EXT_QB")) { try { Disable-ScheduledTask -TaskName $t -ErrorAction Stop | Out-Null } catch {}; try { Stop-ScheduledTask -TaskName $t -ErrorAction Stop } catch {} }
$pat = 'ad_v4_rescore_local|ad_v2_retro|ad_v4_extras|ad_v4_final_set|ad_v2_dock|ad_v4_gates|ad_v4_similarity_search|ad_v4_queue|design_disease_candidates|QUEUE_RUNNER|memrun'
$n = 0
Get-CimInstance Win32_Process -Filter "name='vina.exe'" | Where-Object { $_.CommandLine -match 'Novel Molecular Generation' } | ForEach-Object { & taskkill /T /F /PID $_.ProcessId 2>&1 | Out-Null; $n++ }
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine -match $pat -and $_.ProcessId -ne $PID } | ForEach-Object { & taskkill /T /F /PID $_.ProcessId 2>&1 | Out-Null; $n++ }
Add-Content "logs\queue_control.log" ("[{0}] STOP: {1} processes ended; queue paused and disabled" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $n)
Write-Host "Queue stopped ($n processes ended). Nothing will start again until you run QUEUE_START.ps1. Progress is kept."

