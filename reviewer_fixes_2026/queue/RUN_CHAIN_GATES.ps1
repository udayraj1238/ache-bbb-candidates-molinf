# applies the audited gates (src/ad_v4_gates.py) to every design file as soon as it exists: full-pool chunks A-E and baseline arms
Set-Location -Path $PSScriptRoot
$PY = "python"
$jobs = @(
  @("chunk_A","results\ext\pool_chunk_A.csv"), @("arm_B1","results\ext\arm_B1.csv"), @("chunk_B","results\ext\pool_chunk_B.csv"), @("arm_C0","results\ext\arm_C0.csv"),
  @("chunk_C","results\ext\pool_chunk_C.csv"), @("arm_A","results\ext\arm_A.csv"), @("chunk_D","results\ext\pool_chunk_D.csv"), @("arm_C1","results\ext\arm_C1.csv"),
  @("chunk_E","results\ext\pool_chunk_E.csv"), @("arm_C2","results\ext\arm_C2.csv")
)
function Pending { $jobs | Where-Object { -not (Test-Path ("results\ext\gated_" + $_[0] + "_gated.csv")) } }
while (@(Pending).Count -gt 0) {
    $ran = $false
    foreach ($j in @(Pending)) {
        $gname = $j[0]; $gfile = $j[1]
        if (Test-Path $gfile) {
            $p = Start-Process -FilePath $PY -ArgumentList @("-u","-W","ignore","src\ad_v4_gates.py",("results\ext\gated_" + $gname),$gfile) -PassThru -NoNewWindow -RedirectStandardOutput ("logs\gates_" + $gname + ".out") -RedirectStandardError ("logs\gates_" + $gname + ".err")
            try { $p.PriorityClass = "Idle" } catch {}
            $p.WaitForExit(); $ran = $true; break
        }
    }
    if (-not $ran) { Start-Sleep -Seconds 60 }
}
