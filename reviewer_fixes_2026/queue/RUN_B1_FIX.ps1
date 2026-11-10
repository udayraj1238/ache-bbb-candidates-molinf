# the first B1 chunk (A) was lost when a test run of the chain was cut; after the chain reports completion run it again (chunks that exist are skipped), then gate chunk A
Set-Location -Path $PSScriptRoot
while (-not (Select-String -Path "logs\ext_pool_b1.log" -Pattern "all b1 chunks done" -Quiet)) { Start-Sleep -Seconds 120 }
if (-not (Test-Path "results\ext\b1pool_chunk_A.csv")) { & powershell -NoProfile -ExecutionPolicy Bypass -File .\RUN_EXT_POOL_B1.ps1 }
