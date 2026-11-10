# Baseline arms (items 5, 23, 3): same generator, same 4,000 round-2 seeds, no scaffold cap, different edit libraries.
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
$LOG = "logs\ext_arms.log"
$PY = "python"
function Say($m) { $l = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m; Add-Content -Path $LOG -Value $l -Encoding utf8 }
$arms = @(
  @("B1", "results\ext\lib_B1_ache_preserving.csv", "all"),
  @("C0", "results\ext\lib_C0_random_unused.csv", "all"),
  @("A",  "results\mmp_transform_library_v2.csv", "bbb"),
  @("C1", "results\ext\lib_C1_random_unused.csv", "all"),
  @("C2", "results\ext\lib_C2_random_unused.csv", "all")
)
foreach ($a in $arms) {
    $name, $lib, $mode = $a
    $out = "results\ext\arm_$name.csv"
    if (Test-Path $out) { Say "arm $name exists, skipping"; continue }
    Say "arm $name start ($lib, $mode)"
    $p = Start-Process -FilePath $PY -ArgumentList @("-u","src\design_disease_candidates.py","--disease","alzheimers","--round2-seeds","4000","--max-per-scaffold","100000","--library",$lib,"--library-mode",$mode,"--out",$out) -PassThru -NoNewWindow -RedirectStandardOutput "logs\ext_arm_$name.out" -RedirectStandardError "logs\ext_arm_$name.err"
    try { $p.PriorityClass = "BelowNormal" } catch {}
    $p.WaitForExit()
    Say "arm $name finished, exit $($p.ExitCode)"
}
Say "all arms done"
