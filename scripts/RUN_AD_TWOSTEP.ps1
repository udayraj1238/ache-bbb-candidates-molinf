# ============================================================================
#  Alzheimer's de novo design WITH two-step (double-edit) analogs.
#
#  The first AD pool (859 molecules, RUN_DISEASE_DENOVO.ps1) used
#  --no-two-step, for exactly the reason the first glioma pool did: three
#  diseases were still in play and single-edit analogs already exceeded the
#  100/disease target, so two-step's extra fragmentation cost was not worth
#  spending on arms that would not be carried forward.
#
#  Alzheimer's is now the arm being carried forward, and the single-edit pool
#  has a measured problem that two-step exists to fix. Across the 83 molecules
#  that survive src/ad_hard_gates.py, median Tanimoto similarity to the
#  nearest published AChE active is 0.81, and only 6% fall below 0.70. Those
#  are novel compounds -- absent from PubChem -- but novelty is not
#  non-obviousness, and a single known transformation applied to a published
#  inhibitor is the fact pattern an examiner reads as obvious. A second edit
#  puts a candidate one step further out while every individual edit remains
#  one of the same evidence-scored, real-matched-pair transformations mined
#  in Stage 1b.
#
#  Launched through Task Scheduler rather than Start-Process from an agent
#  session: a session-owned process dies when the host application updates, which
#  has cost this project a run before.
#
#      powershell -ExecutionPolicy Bypass -File .\RUN_AD_TWOSTEP.ps1
#
#  OUTPUT   results/disease_denovo_alzheimers_twostep.csv
#  LOG      logs/ad_twostep.log
# ============================================================================
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$LOG = "logs\ad_twostep.log"

function Say($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path $LOG -Value $line -Encoding utf8
}

$PY = "python"
if (-not (Test-Path $PY)) { $PY = ".\.venv\Scripts\python.exe" }
if (-not (Test-Path $PY)) { $PY = "python" }

Say "=== Alzheimer's two-step design started (PID $PID) ==="
& $PY -u src\design_disease_candidates.py --disease alzheimers `
    --out results\disease_denovo_alzheimers_twostep.csv 2>&1 |
    Tee-Object -FilePath $LOG -Append
Say "=== Alzheimer's two-step design finished ==="
