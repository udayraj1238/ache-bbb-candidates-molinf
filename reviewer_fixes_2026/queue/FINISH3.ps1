# Unattended follow-up of the REINVENT replicate run on Kaggle (second random draw of the same Mol2Mol baseline).  Polls every 15 minutes; runs once, then stops.
Set-Location -Path $PSScriptRoot
$mx = New-Object System.Threading.Mutex($false, "Global\ache_finish3"); if (-not $mx.WaitOne(0)) { exit }
if (Test-Path "results\auto2\rep2") { exit }
$env:PYTHONUTF8 = "1"; $K = "kaggle"; $PY = "python"
function Say($m) { Add-Content -Path "logs\finish3.log" -Value ("[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m) -Encoding utf8 }
function Py($name, $pyargs) { $p = Start-Process -FilePath $PY -ArgumentList (@("-u","-W","ignore") + $pyargs) -PassThru -NoNewWindow -RedirectStandardOutput ("logs\" + $name + ".out") -RedirectStandardError ("logs\" + $name + ".err"); $null = $p.Handle; try { $p.PriorityClass = "Idle"; $p.ProcessorAffinity = [IntPtr]0xF00 } catch {}; $p.WaitForExit(); $p.ExitCode }
Say "finish3 started"
$first = (Get-FileHash "kaggle_reinvent\out\mol2mol_samples_0.csv" -Algorithm MD5).Hash
$t0 = Get-Date
while (((Get-Date) - $t0).TotalHours -lt 60) {
    try {
        $o = (& $K kernels status udayraj1238/ache-reinvent 2>&1 | Out-String)
        if ($o -match "COMPLETE") {
            Remove-Item "kaggle_reinvent\out_rep2" -Recurse -Force -ErrorAction SilentlyContinue; New-Item -ItemType Directory -Force -Path "kaggle_reinvent\out_rep2" | Out-Null
            & $K kernels output udayraj1238/ache-reinvent -p "kaggle_reinvent\out_rep2" 2>&1 | Out-Null
            $f = Get-ChildItem "kaggle_reinvent\out_rep2" -Recurse -Filter "mol2mol_samples_0.csv" | Select-Object -First 1
            if ($f -and (Get-FileHash $f.FullName -Algorithm MD5).Hash -ne $first) {
                $env:M2M_TAG = "_rep2"
                Py "mol2mol_rep2_convert" @("src\ad_v4_mol2mol_convert.py") | Out-Null
                Py "mol2mol_rep2_prep" @("src\ad_v4_gan_baseline_prep.py","results\ext\mol2mol_raw_rep2.csv","results\ext\mol2mol_designs_rep2.csv") | Out-Null
                if (Test-Path "results\ext\mol2mol_designs_rep2.csv") { Py "mol2mol_rep2_gates" @("src\ad_v4_gates.py","results\ext\gated_mol2mol_rep2","results\ext\mol2mol_designs_rep2.csv") | Out-Null }
                Py "mol2mol_rep2_report" @("src\ad_v4_mol2mol_report.py") | Out-Null
                Set-Content "results\auto2\rep2" (Get-Date -Format s); Say "replicate analysed"; break
            } else { Say "kernel complete but output identical to the first run (old version) or missing" }
        }
    } catch { Say ("error: " + $_.Exception.Message) }
    Start-Sleep -Seconds 900
}
Say "finish3 ended"
