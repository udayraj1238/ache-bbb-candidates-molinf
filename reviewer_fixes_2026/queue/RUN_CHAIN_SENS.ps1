# threshold sensitivity (item 15): audited gates on the first study's 2,700 stored designs with every numeric threshold made stricter / looser by 10 % and 20 %
Set-Location -Path $PSScriptRoot
$PY = "python"
foreach ($r in @("0.1","0.2","-0.1","-0.2")) {
    $tag = "sens_" + $r.Replace("-","m").Replace(".","p")
    if (Test-Path ("results\ext\" + $tag + "_gated.csv")) { continue }
    $p = Start-Process -FilePath $PY -ArgumentList @("-u","-W","ignore","src\ad_v4_gates.py",("results\ext\" + $tag),"--relax",$r,"results\disease_denovo_alzheimers_expanded.csv","results\disease_denovo_alzheimers.csv") -PassThru -NoNewWindow -RedirectStandardOutput ("logs\" + $tag + ".out") -RedirectStandardError ("logs\" + $tag + ".err")
    try { $p.PriorityClass = "Idle" } catch {}
    $p.WaitForExit()
}
