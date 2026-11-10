# sequential modelling jobs (items 8, 9, 10, 11), idle priority, one at a time
Set-Location -Path $PSScriptRoot
$PY = "python"
function Run($name, $pyargs) {
    $p = Start-Process -FilePath $PY -ArgumentList (@("-u","-W","ignore") + $pyargs) -PassThru -NoNewWindow -RedirectStandardOutput "logs\$name.out" -RedirectStandardError "logs\$name.err"
    try { $p.PriorityClass = "Idle" } catch {}
    $p.WaitForExit()
}
Run "bbb_indep" @("src\ad_v4_bbb_independence.py")
Run "exposure" @("src\ad_v4_exposure.py")
Run "herg_a" @("src\ad_v4_herg.py","a")
Run "herg_cnt" @("src\ad_v4_herg.py","cnt_desc")
Run "herg_comb" @("src\ad_v4_herg.py","comb")
