# Long unattended queue (laptop, idle priority, 2 workers, memory-gated by the scripts): alternative docking scorers for the benchmark, then the report.  Resumable; ends 2026-10-28 07:00.
Set-Location -Path $PSScriptRoot
$PY = "python"; $END = [datetime]"2026-10-28 07:00"
function Say($m) { Add-Content -Path "logs\long_queue.log" -Value ("[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m) -Encoding utf8 }
function Job($name, $envs, $pyargs) {
    for ($try = 1; $try -le 6; $try++) {
        if ((Get-Date) -gt $END) { return }
        foreach ($k in $envs.Keys) { Set-Item -Path ("Env:" + $k) -Value $envs[$k] }
        Say "$name try $try"
        $p = Start-Process -FilePath $PY -ArgumentList (@("-u","-W","ignore") + $pyargs) -PassThru -NoNewWindow -RedirectStandardOutput ("logs\" + $name + "_" + $try + ".out") -RedirectStandardError ("logs\" + $name + "_" + $try + ".err")
        try { $p.PriorityClass = "Idle" } catch {}
        while (-not $p.HasExited) { if ((Get-Date) -gt $END) { try { $p.Kill() } catch {}; return }; Start-Sleep -Seconds 60 }
        if ((Select-String -Path ("logs\" + $name + "_" + $try + ".out") -Pattern "^done" -Quiet)) { Say "$name finished"; return }
        Start-Sleep -Seconds 300
    }
}
Job "rescore_vinardo" @{VINA_SCORING="vinardo"} @("src\ad_v4_rescore_local.py","vinardo","8","2")
Remove-Item Env:VINA_SCORING -ErrorAction SilentlyContinue
Job "rescore_vina_e32" @{} @("src\ad_v4_rescore_local.py","vina_e32","32","2")
& $PY -u src\ad_v4_rescore_report.py *> "logs\rescore_report.out"
Say "queue finished"
