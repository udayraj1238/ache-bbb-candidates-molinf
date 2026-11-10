# audited gates for the AChE-informed pool chunks in whatever order they appear
Set-Location -Path $PSScriptRoot
$PY = "python"
$names = @("A","B","C","D","E")
function Left { $names | Where-Object { -not (Test-Path ("results\ext\gated_b1chunk_" + $_ + "_gated.csv")) } }
while (@(Left).Count -gt 0) {
    $ran = $false
    foreach ($n in @(Left)) {
        $in = "results\ext\b1pool_chunk_" + $n + ".csv"
        if (Test-Path $in) {
            $p = Start-Process -FilePath $PY -ArgumentList @("-u","-W","ignore","src\ad_v4_gates.py",("results\ext\gated_b1chunk_" + $n),$in) -PassThru -NoNewWindow -RedirectStandardOutput ("logs\gates_b1chunk_" + $n + ".out") -RedirectStandardError ("logs\gates_b1chunk_" + $n + ".err")
            try { $p.PriorityClass = "Idle" } catch {}
            $p.WaitForExit(); $ran = $true; break
        }
    }
    if (-not $ran) { Start-Sleep -Seconds 90 }
}
