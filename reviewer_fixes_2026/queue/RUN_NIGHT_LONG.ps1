# overnight (until 2026-10-30 06:30): after the GAN baseline, a deeper local retrosynthesis pass (USPTO + ring-breaker, 1000 iterations / 300 s) for extended-set candidates the Kaggle passes did not solve;
# then the GAN report and the final set are rebuilt.  One process, idle priority, resumable.
Set-Location -Path $PSScriptRoot
$END = [datetime]"2026-10-30 06:30"; $PY = "python"; $RPY = "C:\retro\venv\Scripts\python.exe"
function Say($m) { Add-Content -Path "logs\night_long.log" -Value ("[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m) -Encoding utf8 }
$t0 = Get-Date
while (-not (Test-Path "results\ext\gated_gan_gated.csv") -and ((Get-Date) - $t0).TotalHours -lt 4) { Start-Sleep -Seconds 120 }
Say "GAN chain done or 4 h passed; starting deep local retrosynthesis"
$env:RETRO_EXP = "uspto,ringbreaker"
for ($try = 1; $try -le 8 -and (Get-Date) -lt $END; $try++) {
    $p = Start-Process -FilePath $RPY -ArgumentList @("-u","-W","ignore","src\ad_v2_retro.py","--in","results\ext\deep_local_todo.csv","--out","results\ext\deep_local_retro.csv","--config","C:/retro/config_deep_ring.yml","--skip-controls") -PassThru -NoNewWindow -RedirectStandardOutput ("logs\deep_local_" + $try + ".out") -RedirectStandardError ("logs\deep_local_" + $try + ".err")
    try { $p.PriorityClass = "Idle" } catch {}
    while (-not $p.HasExited) { if ((Get-Date) -gt $END) { try { $p.Kill() } catch {}; break }; Start-Sleep -Seconds 60 }
    if ($p.HasExited -and $p.ExitCode -eq 0) { Say "deep local retro finished"; break }
    Say "deep local retro try $try ended"; Start-Sleep -Seconds 120
}
& $PY -u -W ignore src\ad_v4_gan_report.py *> "logs\night_gan_report.out"
& $PY -u -W ignore src\ad_v4_final_set.py *> "logs\night_final_set.out"
Say "night queue finished"
