# ============================================================================
#  Alzheimer's design, expanded: 16,000 second-edit seeds instead of 4,000.
#
#  WHY
#  The supervisor's brief asks for at least 100 molecules. The anchored hard
#  gates deliver 77 from the 4,000-seed run. There are two ways to close that
#  gap and only one is defensible:
#
#    * lower a threshold  -- would reach 100 in minutes, and is exactly the
#      "choose the bar after seeing what it produces" error this project was
#      written to criticise. Not done.
#    * design more molecules -- costs compute and moves no threshold at all.
#      This script.
#
#  The lever is the round-2 seed cap, hardcoded at 4,000 back when three
#  diseases shared one deadline: only 4,000 of 38,209 round-1 analogs were
#  ever used as starting points for a second edit. Two-edit molecules are
#  the ones most worth having on this arm -- 100% novel in PubChem against
#  81% for single-edit, median 0.723 similarity to prior art against 0.825,
#  and a slightly BETTER median docking score. Raising the cap makes more of
#  exactly the molecules the portfolio wants, and every gate downstream
#  stays where it is.
#
#  COST: roughly 4x the design time (~90-110 min) and ~1.7 GB peak. Nothing
#  downstream changes.
#
#  Launched through Task Scheduler, not Start-Process from an agent session:
#  a session-owned process dies when the host application updates.
#
#      powershell -ExecutionPolicy Bypass -File .\RUN_AD_EXPAND.ps1
#
#  OUTPUT   results/disease_denovo_alzheimers_expanded.csv
#  LOG      logs/ad_expand.log
# ============================================================================
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$LOG = "logs\ad_expand.log"

function Say($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path $LOG -Value $line -Encoding utf8
}

$PY = "python"
if (-not (Test-Path $PY)) { $PY = ".\.venv\Scripts\python.exe" }
if (-not (Test-Path $PY)) { $PY = "python" }

Say "=== Alzheimer's expanded design started (16000 round-2 seeds, PID $PID) ==="
& $PY -u src\design_disease_candidates.py --disease alzheimers `
    --round2-seeds 16000 `
    --out results\disease_denovo_alzheimers_expanded.csv 2>&1 |
    Tee-Object -FilePath $LOG -Append
Say "=== Alzheimer's expanded design finished ==="
