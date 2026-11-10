# Unattended follow-up for the Kaggle jobs of items 10 (ChemBERTa embeddings -> hERG views) and 23 (REINVENT 4 Mol2Mol baseline).  Polls every 15 min (tiny footprint), idle priority for the local steps.
Set-Location -Path $PSScriptRoot
$mx = New-Object System.Threading.Mutex($false, "Global\ache_finish2"); if (-not $mx.WaitOne(0)) { exit }
$env:PYTHONUTF8 = "1"; $K = "kaggle"; $PY = "python"
New-Item -ItemType Directory -Force -Path "results\auto2" | Out-Null
function Say($m) { Add-Content -Path "logs\finish2.log" -Value ("[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m) -Encoding utf8 }
function Done($n) { Test-Path ("results\auto2\" + $n) }
function Mark($n) { Set-Content -Path ("results\auto2\" + $n) -Value (Get-Date -Format s) }
function Status($slug) { $o = (& $K kernels status ("udayraj1238/" + $slug) 2>&1 | Out-String); if ($o -match "COMPLETE") { "COMPLETE" } elseif ($o -match "ERROR|CANCEL") { "FAILED" } else { "OTHER" } }
function Py($name, $pyargs) { $p = Start-Process -FilePath $PY -ArgumentList (@("-u","-W","ignore") + $pyargs) -PassThru -NoNewWindow -RedirectStandardOutput ("logs\" + $name + ".out") -RedirectStandardError ("logs\" + $name + ".err"); try { $p.PriorityClass = "Idle" } catch {}; $p.WaitForExit(); $p.ExitCode }
function Pull($slug, $dir) { Remove-Item $dir -Recurse -Force -ErrorAction SilentlyContinue; New-Item -ItemType Directory -Force -Path $dir | Out-Null; & $K kernels output ("udayraj1238/" + $slug) -p $dir 2>&1 | Out-Null }
Say "finish2 started"; $t0 = Get-Date
while (((Get-Date) - $t0).TotalHours -lt 72) {
    try {
        if (-not (Done "embed")) {
            $s = Status "ache-embed"
            if ($s -eq "COMPLETE") {
                Pull "ache-embed" "kaggle_embed\out"; $f = Get-ChildItem "kaggle_embed\out" -Recurse -Filter "herg_chemberta.npz" | Select -First 1
                if ($f) { Copy-Item $f.FullName "results\ext\herg_chemberta.npz" -Force; Py "herg_chem" @("src\ad_v4_herg.py","chem") | Out-Null; Py "herg_chem_a" @("src\ad_v4_herg.py","chem_a") | Out-Null; Mark "embed"; Say "hERG chem views done" }
                else { Mark "embed"; Say "embed kernel complete but no npz" }
            } elseif ($s -eq "FAILED") { Mark "embed"; Say "embed kernel failed" }
        }
        if (-not (Done "reinvent")) {
            $s = Status "ache-reinvent"
            if ($s -eq "COMPLETE") {
                Pull "ache-reinvent" "kaggle_reinvent\out"
                if ((Get-ChildItem "kaggle_reinvent\out" -Recurse -Filter "mol2mol_samples*.csv").Count -gt 0) {
                    Py "mol2mol_convert" @("src\ad_v4_mol2mol_convert.py") | Out-Null
                    Py "mol2mol_prep" @("src\ad_v4_gan_baseline_prep.py","results\ext\mol2mol_raw.csv","results\ext\mol2mol_designs.csv") | Out-Null
                    if (Test-Path "results\ext\mol2mol_designs.csv") { Py "mol2mol_gates" @("src\ad_v4_gates.py","results\ext\gated_mol2mol","results\ext\mol2mol_designs.csv") | Out-Null }
                    Get-Content logs\mol2mol_convert.out, logs\mol2mol_prep.out | Set-Content logs\mol2mol_all.out
                    Py "mol2mol_report" @("src\ad_v4_mol2mol_report.py") | Out-Null; Mark "reinvent"; Say "mol2mol baseline done"
                } else { Say "reinvent kernel complete but no samples yet (an older version, or a failure: see kaggle_reinvent\out log)" }
            } elseif ($s -eq "FAILED") { Say "reinvent kernel in error state (may be an older version; retrying)" }
        }
        if ((Done "embed") -and (Done "reinvent")) { break }
    } catch { Say ("error: " + $_.Exception.Message) }
    Start-Sleep -Seconds 900
}
Say "finish2 ended"



