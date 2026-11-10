# Full-pool design (item 14): every round-1 analogue and every round-2 seed (38,209), no per-scaffold cap, in five chunks to bound memory.
# Output: results/ext/pool_chunk_<A..E>.csv   Log: logs/ext_pool.log      Launched through Task Scheduler (see memory note).
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
New-Item -ItemType Directory -Force -Path "logs","results\ext" | Out-Null
$LOG = "logs\ext_pool.log"
$PY = "python"
function Say($m) { $l = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m; Add-Content -Path $LOG -Value $l -Encoding utf8 }
$chunks = @(@("A",0,8000), @("B",8000,16000), @("C",16000,24000), @("D",24000,32000), @("E",32000,40000))
foreach ($c in $chunks) {
    $name, $s, $e = $c
    $out = "results\ext\pool_chunk_$name.csv"
    if (Test-Path $out) { Say "chunk $name exists, skipping"; continue }
    Say "chunk $name seeds [$s,$e) start"
    $p = Start-Process -FilePath $PY -ArgumentList @("-u","src\design_disease_candidates.py","--disease","alzheimers","--round2-seeds","$e","--round2-start","$s","--max-per-scaffold","100000","--out",$out) -PassThru -NoNewWindow -RedirectStandardOutput "logs\ext_pool_$name.out" -RedirectStandardError "logs\ext_pool_$name.err"
    try { $p.PriorityClass = "BelowNormal" } catch {}
    $p.WaitForExit()
    Say "chunk $name finished, exit $($p.ExitCode)"
}
Say "all chunks done"
