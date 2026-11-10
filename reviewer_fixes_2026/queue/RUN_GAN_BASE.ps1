# item 23: GAN baseline (sample -> design-stage columns -> audited gates), one job at a time, idle priority
Set-Location -Path $PSScriptRoot
$PY = "python"
function Run($name, $pyargs, $doneFile) {
    for ($i = 1; $i -le 4; $i++) {
        if (Test-Path $doneFile) { return }
        $p = Start-Process -FilePath $PY -ArgumentList (@("-u","-W","ignore") + $pyargs) -PassThru -NoNewWindow -RedirectStandardOutput ("logs\" + $name + "_" + $i + ".out") -RedirectStandardError ("logs\" + $name + "_" + $i + ".err")
        try { $p.PriorityClass = "Idle" } catch {}
        $p.WaitForExit(); if (Test-Path $doneFile) { return }; Start-Sleep -Seconds 120
    }
}
$env:GAN_NODES = "38"
Run "gan_sample" @("src\ad_v4_gan_baseline_sample.py","30000") "results\ext\gan_baseline_raw.csv"
if (Test-Path "results\ext\gan_baseline_raw.csv") { $r = (Get-Item "results\ext\gan_baseline_raw.csv").Length }
Run "gan_prep" @("src\ad_v4_gan_baseline_prep.py") "results\ext\gan_baseline_designs.csv"
Run "gan_gates" @("src\ad_v4_gates.py","results\ext\gated_gan","results\ext\gan_baseline_designs.csv") "results\ext\gated_gan_gated.csv"
