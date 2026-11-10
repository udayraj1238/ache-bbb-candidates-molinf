# Starts (or restarts) the queue.  Safe to run at any time and as often as you like: finished jobs are skipped, unfinished ones resume from their saved results, and only one runner per lane can exist.
Set-Location -Path $PSScriptRoot
Remove-Item "queue\PAUSE" -ErrorAction SilentlyContinue
foreach ($t in @("EXT_QA","EXT_QB")) { try { Enable-ScheduledTask -TaskName $t -ErrorAction Stop | Out-Null } catch {}; try { Start-ScheduledTask -TaskName $t -ErrorAction Stop } catch {} }
Add-Content "logs\queue_control.log" ("[{0}] START" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
Write-Host "Queue started (both lanes). Check progress with QUEUE_STATUS.ps1."
