# item 24: second target (BChE): design from potent BChE inhibitors with the same generator and the same audited gates; time-split validation of its activity model
Set-Location -Path $PSScriptRoot
$PY = "python"
function Run($name, $pyargs, $prio) {
    $p = Start-Process -FilePath $PY -ArgumentList (@("-u","-W","ignore") + $pyargs) -PassThru -NoNewWindow -RedirectStandardOutput ("logs\" + $name + ".out") -RedirectStandardError ("logs\" + $name + ".err")
    try { $p.PriorityClass = $prio } catch {}
    $p.WaitForExit()
}
if (-not (Test-Path "results\ext\bche_arm.csv")) { Run "bche_design" @("src\memrun.py","1.3","src\design_disease_candidates.py","--disease","bche","--round2-seeds","4000","--max-per-scaffold","100000","--out","results\ext\bche_arm.csv") "BelowNormal" }
if (-not (Test-Path "results\ext\gated_bche_gated.csv")) { Run "gates_bche" @("src\ad_v4_gates.py","results\ext\gated_bche","results\ext\bche_arm.csv","--target","bche") "Idle" }
$env:ACT_TARGET = "bche"
Run "bche_fit" @("src\ad_v4_activity_fit.py","fit") "Idle"
Run "bche_eval" @("src\ad_v4_activity_eval.py") "Idle"
