# Unattended finisher: polls Kaggle every 10 min (tiny footprint), downloads finished kernels, pushes the follow-up kernels, runs the analysis scripts, and finally commits to the repo.  No Claude session needed.
Set-Location -Path $PSScriptRoot
$env:PYTHONUTF8 = "1"
$K = "kaggle"
$PY = "python"
New-Item -ItemType Directory -Force -Path "results\auto" | Out-Null
function Say($m) { Add-Content -Path "logs\finish_auto.log" -Value ("[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m) -Encoding utf8 }
function Done($n) { Test-Path ("results\auto\" + $n) }
function Mark($n) { Set-Content -Path ("results\auto\" + $n) -Value (Get-Date -Format s) }
function Status($slug) { $o = (& $K kernels status ("udayraj1238/" + $slug) 2>&1 | Out-String); if ($o -match "COMPLETE") { "COMPLETE" } elseif ($o -match "ERROR|CANCEL") { "FAILED" } else { "OTHER" } }
function Py($name, $pyargs) { $p = Start-Process -FilePath $PY -ArgumentList (@("-u","-W","ignore") + $pyargs) -PassThru -NoNewWindow -RedirectStandardOutput ("logs\" + $name + ".out") -RedirectStandardError ("logs\" + $name + ".err"); try { $p.PriorityClass = "Idle" } catch {}; $p.WaitForExit(); $p.ExitCode }
function Pull($slug, $dir) { New-Item -ItemType Directory -Force -Path $dir | Out-Null; & $K kernels output ("udayraj1238/" + $slug) -p $dir 2>&1 | Out-Null; (Get-ChildItem $dir -Recurse -File | Measure-Object).Count -gt 0 }
Say "finisher started"
$t0 = Get-Date
while (((Get-Date) - $t0).TotalHours -lt 60) {
    try {
        # 1 retrosynthesis of the extras -> download, then start the second planner
        if (-not (Done "retro_extras") -and (Status "ache-retro-extras") -eq "COMPLETE") { if (Pull "ache-retro-extras" "kaggle_retro\out_extras") { Mark "retro_extras"; Say "retro-extras downloaded" } }
        if ((Done "retro_extras") -and -not (Done "second_pushed")) {
            $o = (& $K kernels push -p "kaggle_retro\kernel_ache-retro-second" 2>&1 | Out-String); Say ("push second: " + $o.Trim())
            if ($o -match "successfully") { Mark "second_pushed" }
        }
        if ((Done "second_pushed") -and -not (Done "second_done") -and (Status "ache-retro-second") -eq "COMPLETE") {
            Remove-Item "kaggle_retro\out_second" -Recurse -Force -ErrorAction SilentlyContinue; if ((Pull "ache-retro-second" "kaggle_retro\out_second") -and (Test-Path "kaggle_retro\out_second\out") -and ((Get-ChildItem "kaggle_retro\out_second" -Recurse -Filter "retro_second_s*.csv").Count -gt 0)) { Py "route_second" @("src\ad_v4_route_second_summary.py") | Out-Null; Mark "second_done"; Say "second planner analysed" }
        }
        # 2 MD batches
        if (-not (Done "md_done") -and (Status "ache-md3-kb") -eq "COMPLETE" -and (Status "ache-md3-kc") -eq "COMPLETE") {
            if ((Pull "ache-md3-kb" "kaggle_md3\out_kb") -and (Pull "ache-md3-kc" "kaggle_md3\out_kc")) { Py "md_analysis" @("src\ad_v4_md_analysis.py") | Out-Null; Mark "md_done"; Say "MD analysed" }
        }
        # 3 ensemble docking
        if (-not (Done "ens_done")) {
            $all = $true
            foreach ($r in @("4ey4","4ey5","4ey6","4m0e")) {
                $d = "kaggle_ens\out2_" + $r.ToUpper()
                if (-not (Test-Path ($d + "\out"))) { if ((Status ("ache-ens-" + $r)) -eq "COMPLETE") { Pull ("ache-ens-" + $r) $d | Out-Null; $c = Get-ChildItem $d -Recurse -Filter "ens_*.csv" | Select -First 1; if (-not $c -or (Get-Content $c.FullName).Count -gt 130) { Remove-Item $d -Recurse -Force -ErrorAction SilentlyContinue } } }
                if (-not (Test-Path ($d + "\out"))) { $all = $false }
            }
            if ($all) { Py "ens_report" @("src\ad_v4_ens_report.py") | Out-Null; Mark "ens_done"; Say "ensemble docking analysed" }
        }
        # 4 round 2 (AChE-informed library pool) -> retrosynthesis of its new molecules
        if ((Test-Path "results\ext\extras2_toretro.csv") -and -not (Done "sim2")) {
            if ((Py "sim_extras2" @("src\ad_v4_similarity_search.py","results\ext\extras2_toretro.csv","results\AD_V4_SIMILARITY_SEARCH_EXTRAS2.csv")) -eq 0) { Mark "sim2"; Say "similarity search for round 2 done" }
        }
        if ((Test-Path "results\ext\extras2_toretro.csv") -and -not (Done "retro2_pushed")) {
            $d = "kaggle_retro\dataset2"; New-Item -ItemType Directory -Force -Path $d | Out-Null
            Copy-Item "results\ext\extras2_toretro.csv" "$d\extras2.csv" -Force; Copy-Item "kaggle_retro\dataset\retro_run.py" "$d\retro_run.py" -Force
            Set-Content "$d\dataset-metadata.json" -Value '{"title": "ache-retro-inputs2", "id": "udayraj1238/ache-retro-inputs2", "licenses": [{"name": "CC0-1.0"}], "isPrivate": true}' -Encoding ascii
            $o = (& $K datasets create -p $d --dir-mode zip 2>&1 | Out-String); Say ("dataset2: " + $o.Trim())
            if ($o -match "successfully|already") {
                Start-Sleep -Seconds 90
                Push-Location kaggle_retro; & $PY make_nb.py first extras2.csv ache-retro-extras2 200 120 ache-retro-inputs2; Pop-Location
                $o2 = (& $K kernels push -p "kaggle_retro\kernel_ache-retro-extras2" 2>&1 | Out-String); Say ("push extras2: " + $o2.Trim())
                if ($o2 -match "successfully") { Mark "retro2_pushed" }
            }
        }
        if ((Done "retro2_pushed") -and -not (Done "retro2_done") -and (Status "ache-retro-extras2") -eq "COMPLETE") { if (Pull "ache-retro-extras2" "kaggle_retro\out_extras2") { Mark "retro2_done"; Say "retro-extras2 downloaded" } }
        # deep retrosynthesis pass (1000 iterations, 300 s) for the extras the first pass did not solve
        if ((Done "retro_extras") -and -not (Done "deep_pushed")) {
            $o = (& $K kernels push -p "kaggle_retro\kernel_ache-retro-deep" 2>&1 | Out-String); Say ("push deep: " + $o.Trim()); if ($o -match "successfully") { Mark "deep_pushed" }
        }
        if ((Done "deep_pushed") -and -not (Done "deep_done") -and (Status "ache-retro-deep") -eq "COMPLETE") {
            Remove-Item "kaggle_retro\out_extras_deep" -Recurse -Force -ErrorAction SilentlyContinue
            if ((Pull "ache-retro-deep" "kaggle_retro\out_extras_deep") -and ((Get-ChildItem "kaggle_retro\out_extras_deep" -Recurse -Filter "retro_first_s*.csv").Count -gt 0)) { Mark "deep_done"; Say "deep retro downloaded" }
        }
        # GNINA CNN rescoring (GPU kernel)
        if (-not (Done "gnina_done") -and (Status "ache-gnina") -eq "COMPLETE") {
            Remove-Item "kaggle_ens\out_gnina" -Recurse -Force -ErrorAction SilentlyContinue
            if ((Pull "ache-gnina" "kaggle_ens\out_gnina") -and ((Get-ChildItem "kaggle_ens\out_gnina" -Recurse -Filter "gnina_4EY7.csv").Count -gt 0)) { Py "gnina_report" @("src\ad_v4_gnina_report.py") | Out-Null; Mark "gnina_done"; Say "gnina analysed" }
        }
        # 5 final assembly
        $r2 = (((Done "retro2_done") -and (Done "deep_done")) -or ((Get-Date) -gt [datetime]"2026-10-26 23:40"))
        if ((Done "retro_extras") -and $r2 -and ((Done "sim2") -or -not (Test-Path "results\ext\extras2_toretro.csv")) -and -not (Done "final")) {
            Py "pool_report" @("src\ad_v4_pool_report.py") | Out-Null; Py "sens_report" @("src\ad_v4_sensitivity_report.py") | Out-Null; Py "final_set" @("src\ad_v4_final_set.py") | Out-Null
            Mark "final"; Say "final set assembled"
        }
        # 6 everything done -> save to the repo and stop
        if ((Done "final") -and -not (Done "committed") -and (((Done "gnina_done") -and (Test-Path "results\AD_V4_RESCORE_REPORT.txt") -and (Done "ens_done") -and (Done "second_done")) -or ((Get-Date) -gt [datetime]"2026-10-28 06:00"))) {
            $R = "C:\bbbrep\ache_repo"
            if (Test-Path $R) {
                Copy-Item "src\ad_v4_*.py","src\memgate.py","src\memrun.py","src\design_disease_candidates.py" "$R\src\" -Force -ErrorAction SilentlyContinue
                New-Item -ItemType Directory -Force -Path "$R\results\overnight" | Out-Null
                Copy-Item "results\AD_V4_*","results\PREREG_*","results\OVERNIGHT_STATUS.md" "$R\results\overnight\" -Force -ErrorAction SilentlyContinue
                Push-Location $R; git add -A 2>&1 | Out-Null; git -c user.name=udayraj1238 -c user.email=udayraj1238@users.noreply.github.com commit -m "Overnight analyses for the reviewer-weakness fixes (activity model, edit libraries, full pool, BBB independence, MD v3, ensemble docking, second route planner)" 2>&1 | Out-Null; Pop-Location
                Say "committed to the local repo (not pushed)"
            }
            Mark "committed"; break
        }
    } catch { Say ("error: " + $_.Exception.Message) }
    Start-Sleep -Seconds 600
}
Say "finisher ended"
