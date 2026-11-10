# round 2 of the downstream gates: survivors of the AChE-informed library pool not processed in round 1
Set-Location -Path $PSScriptRoot
$PY = "python"
$env:EXTRAS_TAG = "2"
function Run($name, $pyargs) {
    $p = Start-Process -FilePath $PY -ArgumentList (@("-u","-W","ignore") + $pyargs) -PassThru -NoNewWindow -RedirectStandardOutput ("logs\" + $name + ".out") -RedirectStandardError ("logs\" + $name + ".err")
    try { $p.PriorityClass = "Idle" } catch {}
    $p.WaitForExit()
}
while (-not ((Test-Path "results\ext\gated_b1chunk_A_gated.csv") -and (Test-Path "results\ext\gated_b1chunk_B_gated.csv") -and (Test-Path "results\ext\gated_b1chunk_C_gated.csv") -and (Test-Path "results\ext\gated_b1chunk_D_gated.csv") -and (Test-Path "results\ext\gated_b1chunk_E_gated.csv"))) { Start-Sleep -Seconds 120 }
foreach ($step in @("collect","stereo","offline","novelty","dock","toretro")) { Run ("extras2_" + $step) @("src\ad_v4_extras.py",$step) }
