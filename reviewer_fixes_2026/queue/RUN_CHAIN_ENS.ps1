# waits for the MD-set docking to finish, then runs the ensemble docking (item 12) at Idle priority
Set-Location -Path $PSScriptRoot
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*ad_v4_md_set*" }) { Start-Sleep -Seconds 60 }
$p = Start-Process -FilePath "python" -ArgumentList @("-u","-W","ignore","src\ad_v4_ens_dock.py","2") -PassThru -NoNewWindow -RedirectStandardOutput "logs\ens_dock.out" -RedirectStandardError "logs\ens_dock.err"
try { $p.PriorityClass = "Idle" } catch {}
$p.WaitForExit()
