# Same stereo + novelty + readiness chain for the extra (5-10 % margin) molecules. Low priority; Task Scheduler launch.
Set-Location "<PROJECT_ROOT>"
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'
$env:OPENBLAS_NUM_THREADS = "2"
$py = "python"
function Step($name, $argstr) { "=== $name started $(Get-Date)" | Out-File logs\ad_v2_chainX.log -Append -Encoding utf8; & $py -u "src\$name.py" $argstr.Split(" ") *>> logs\ad_v2_chainX.log; "=== $name finished $(Get-Date) exit $LASTEXITCODE" | Out-File logs\ad_v2_chainX.log -Append -Encoding utf8 }
Step "ad_resolve_stereo" "--in results\AD_V2_X.csv"
$d = & $py -c "import pandas as pd; d=pd.read_csv('results/AD_V2_X_stereo.csv'); d=d[d.stereo_status!='parent_undefined']; d.to_csv('results/AD_V2_X2.csv',index=False); d.to_csv('results/AD_V2_X2_chembl.csv',index=False); d.to_csv('results/AD_V2_X2_surechembl.csv',index=False); print(len(d))"
Step "check_novelty"         "--in results\AD_V2_X2.csv --out results\AD_V2_X2_pubchem.csv"
Step "check_novelty_flex"    "--in results\AD_V2_X2.csv --out results\AD_V2_X2_flex.csv"
Step "check_chembl"          "--in results\AD_V2_X2_chembl.csv"
Step "check_surechembl"      "--in results\AD_V2_X2_surechembl.csv"
Step "ad_synthesis_readiness" "--in results\AD_V2_X2.csv --out results\AD_V2_X2_readiness.csv"
Step "ad_bche_selectivity"   "--in results\AD_V2_X2.csv --out results\AD_V2_X2_bche.csv"
Step "ad_bcrp_flag"          "--in results\AD_V2_X2.csv --out results\AD_V2_X2_bcrp.csv"
"=== ALL DONE $(Get-Date)" | Out-File logs\ad_v2_chainX.log -Append -Encoding utf8
