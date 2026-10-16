# Novelty (PubChem, ChEMBL flexmatch, ChEMBL, SureChEMBL), supplier/route readiness, BChE and BCRP flags for the 142-molecule pool.
# Network-bound, low priority; launched through Task Scheduler.
Set-Location "<PROJECT_ROOT>"
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'
$env:OPENBLAS_NUM_THREADS = "2"
$py = "python"
$steps = @(
 @("check_novelty",        "--in results\AD_V2_P2.csv --out results\AD_V2_P2_pubchem.csv"),
 @("check_novelty_flex",   "--in results\AD_V2_P2.csv --out results\AD_V2_P2_flex.csv"),
 @("check_chembl",         "--in results\AD_V2_P2_chembl.csv"),
 @("check_surechembl",     "--in results\AD_V2_P2_surechembl.csv"),
 @("ad_synthesis_readiness","--in results\AD_V2_P2.csv --out results\AD_V2_P2_readiness.csv"),
 @("ad_bche_selectivity",  "--in results\AD_V2_P2.csv --out results\AD_V2_P2_bche.csv"),
 @("ad_bcrp_flag",         "--in results\AD_V2_P2.csv --out results\AD_V2_P2_bcrp.csv")
)
foreach ($s in $steps) {
  "=== $($s[0]) started $(Get-Date)" | Out-File logs\ad_v2_chain.log -Append -Encoding utf8
  & $py -u "src\$($s[0]).py" $s[1].Split(" ") *>> logs\ad_v2_chain.log
  "=== $($s[0]) finished $(Get-Date) exit $LASTEXITCODE" | Out-File logs\ad_v2_chain.log -Append -Encoding utf8
}
"=== ALL DONE $(Get-Date)" | Out-File logs\ad_v2_chain.log -Append -Encoding utf8
