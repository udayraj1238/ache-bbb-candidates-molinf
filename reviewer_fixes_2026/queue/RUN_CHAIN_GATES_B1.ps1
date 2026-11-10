Set-Location -Path $PSScriptRoot
$PY = "python"
foreach ($n in @("A","B","C","D","E")) {
    $in = "results\ext\b1pool_chunk_$n.csv"; $outp = "results\ext\gated_b1chunk_$n"
    while (-not (Test-Path $in)) { Start-Sleep -Seconds 90 }
    if (Test-Path ($outp + "_gated.csv")) { continue }
    $p = Start-Process -FilePath $PY -ArgumentList @("-u","-W","ignore","src\ad_v4_gates.py",$outp,$in) -PassThru -NoNewWindow -RedirectStandardOutput ("logs\gates_b1chunk_$n.out") -RedirectStandardError ("logs\gates_b1chunk_$n.err")
    try { $p.PriorityClass = "Idle" } catch {}
    $p.WaitForExit()
}
