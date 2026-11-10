# AChE-informed library (68 edits whose potency effect was measured in real AChE pairs) over the full seed pool, uncapped; starts after the BBB-library pool is done
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
$PY = "python"
function Say($m) { $l = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m; Add-Content -Path "logs\ext_pool_b1.log" -Value $l -Encoding utf8 }
while (-not (Select-String -Path "logs\ext_pool.log" -Pattern "all chunks done" -Quiet)) { Start-Sleep -Seconds 120 }
$chunks = @(@("A",0,8000), @("B",8000,16000), @("C",16000,24000), @("D",24000,32000), @("E",32000,40000))
foreach ($c in $chunks) {
    $name = $c[0]; $s = $c[1]; $e = $c[2]; $out = "results\ext\b1pool_chunk_$name.csv"
    if (Test-Path $out) { continue }
    Say "b1 chunk $name start"
    $p = Start-Process -FilePath $PY -ArgumentList @("-u","src\memrun.py","1.4","src\design_disease_candidates.py","--disease","alzheimers","--round2-seeds","$e","--round2-start","$s","--max-per-scaffold","100000","--library","results\ext\lib_B1_ache_preserving.csv","--library-mode","all","--out",$out) -PassThru -NoNewWindow -RedirectStandardOutput "logs\ext_b1pool_$name.out" -RedirectStandardError "logs\ext_b1pool_$name.err"
    try { $p.PriorityClass = "BelowNormal" } catch {}
    $p.WaitForExit(); Say "b1 chunk $name finished"
}
Say "all b1 chunks done"
