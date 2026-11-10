# item 14: downstream gates for the audited-gate survivors of the full pool that the first study never processed (and the 58 removed by the legacy P-gp gate)
Set-Location -Path $PSScriptRoot
$PY = "python"; $RY = "C:\retro\venv\Scripts\python.exe"
function Run($name, $exe, $pyargs, $prio) {
    $p = Start-Process -FilePath $exe -ArgumentList (@("-u","-W","ignore") + $pyargs) -PassThru -NoNewWindow -RedirectStandardOutput ("logs\" + $name + ".out") -RedirectStandardError ("logs\" + $name + ".err")
    try { $p.PriorityClass = $prio } catch {}
    $p.WaitForExit()
}
while (-not ((Test-Path "results\ext\gated_chunk_A_gated.csv") -and (Test-Path "results\ext\gated_chunk_B_gated.csv") -and (Test-Path "results\ext\gated_chunk_C_gated.csv") -and (Test-Path "results\ext\gated_chunk_D_gated.csv") -and (Test-Path "results\ext\gated_chunk_E_gated.csv"))) { Start-Sleep -Seconds 120 }
Run "extras_collect" $PY @("src\ad_v4_extras.py","collect") "Idle"
Run "extras_stereo" $PY @("src\ad_v4_extras.py","stereo") "Idle"
Run "extras_offline" $PY @("src\ad_v4_extras.py","offline") "Idle"
Run "extras_novelty" $PY @("src\ad_v4_extras.py","novelty") "Idle"
Run "extras_dock" $PY @("src\ad_v4_extras.py","dock") "Idle"
Run "extras_toretro" $PY @("src\ad_v4_extras.py","toretro") "Idle"
Run "extras_retro" $RY @("src\memrun.py","1.4","src\ad_v2_retro.py","--in","results\ext\extras_toretro.csv","--out","results\ext\extras_retro_s0.csv") "Idle"
