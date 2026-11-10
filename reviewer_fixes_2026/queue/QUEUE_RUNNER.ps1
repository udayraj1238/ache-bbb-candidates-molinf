# Unattended queue lane.  Usage: QUEUE_RUNNER.ps1 -Lane A|B     (started by the scheduled tasks EXT_QA / EXT_QB every 10 minutes; a named mutex lets only one instance per lane run)
# Built to share the laptop with other projects:
# - every job is resumable (progress lives in its own output files), so a kill, sleep or restart costs at most the molecule in flight;
# - jobs run at idle priority; the lane SUSPENDS its processes (no loss) while the whole CPU is above 92 % and resumes below 70 %;
# - the lane waits for free memory before a job and KILLS its job (progress kept) if free memory stays under 0.35 GB for a minute while it runs, so it never takes memory the other projects need;
# - a job killed from outside (another project's cleanup) is simply started again; a job that fails is retried after a 30-minute cooldown, never dropped for good;
# - queue\PAUSE (created by QUEUE_STOP.ps1) pauses the lane; at the end time the at_end commands run once.
param([string]$Lane = "A")
Set-Location -Path $PSScriptRoot
$END = [datetime]"2026-11-04 23:00"
$mutex = New-Object System.Threading.Mutex($false, "Global\ache_queue_$Lane"); if (-not $mutex.WaitOne(0)) { exit }
if ((Get-Date) -gt $END -and (Test-Path "queue\end_done_$Lane")) { exit }
$PY = "python"; $RPY = "C:\retro\venv\Scripts\python.exe"
$log = "logs\queue_$Lane.log"; $doneDir = "queue\done_$Lane"; New-Item -ItemType Directory -Force -Path $doneDir | Out-Null
Add-Type -Namespace Q -Name Nt -MemberDefinition '[DllImport("ntdll.dll")] public static extern int NtSuspendProcess(IntPtr h); [DllImport("ntdll.dll")] public static extern int NtResumeProcess(IntPtr h);'
function Say($m) { Add-Content -Path $log -Value ("[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m) -Encoding utf8 }
function FreeGB { [math]::Round((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory / 1MB, 1) }
function CpuLoad { try { [int](Get-CimInstance Win32_Processor | Measure-Object LoadPercentage -Average).Average } catch { 0 } }
function Tree($rootId) { $all = Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId; $ids = @($rootId); $q = @($rootId); while ($q.Count) { $n = @(); foreach ($x in $q) { $n += @($all | Where-Object { $_.ParentProcessId -eq $x } | ForEach-Object { $_.ProcessId }) }; $ids += $n; $q = $n }; $ids }
function KillTree($p) { try { & taskkill /T /F /PID $p.Id 2>&1 | Out-Null } catch {} }
function Suspend($p, $on) { foreach ($i in (Tree $p.Id)) { try { $h = (Get-Process -Id $i -ErrorAction Stop).Handle; if ($on) { [Q.Nt]::NtSuspendProcess($h) | Out-Null } else { [Q.Nt]::NtResumeProcess($h) | Out-Null } } catch {} } }
function Limit-Tree($p) { $mask = if ($Lane -eq "A") { 0x0F00 } else { 0xF000 }; foreach ($i in (Tree $p.Id)) { try { $q = Get-Process -Id $i -ErrorAction Stop; if ($q.PriorityClass -ne "Idle") { $q.PriorityClass = "Idle" }; if ([int64]$q.ProcessorAffinity -ne $mask) { $q.ProcessorAffinity = [IntPtr]$mask } } catch {} } }
function Paused { Test-Path "queue\PAUSE" }
. ".\queue\jobs_$Lane.ps1"
Say "lane $Lane started; $($jobs.Count) jobs; ends $END"
# if an earlier runner died while it had its jobs suspended (CPU throttle), wake them up again
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'ad_v4_rescore_local|ad_v2_retro|ad_v4_extras|ad_v4_final_set|ad_v2_dock|ad_v4_gates|ad_v4_similarity_search|design_disease_candidates' -or ($_.Name -eq 'vina.exe' -and $_.CommandLine -match 'docking') } | ForEach-Object { try { [Q.Nt]::NtResumeProcess((Get-Process -Id $_.ProcessId -ErrorAction Stop).Handle) | Out-Null } catch {} }
while ((Get-Date) -lt $END) {
    $ran = $false
    foreach ($j in $jobs) {
        if ((Get-Date) -gt $END) { break }
        if (Test-Path "$doneDir\$($j.name)") { continue }
        if ($j.after -and -not (Test-Path "$doneDir\$($j.after)")) { continue }
        $fm = "$doneDir\$($j.name).failed"
        if (Test-Path $fm) { if (((Get-Date) - (Get-Item $fm).LastWriteTime).TotalMinutes -lt 30) { continue } else { Remove-Item $fm -ErrorAction SilentlyContinue } }
        $ok = $false
        for ($try = 1; $try -le $j.tries -and (Get-Date) -lt $END -and -not $ok; $try++) {
            while (Paused) { Start-Sleep -Seconds 30; if ((Get-Date) -gt $END) { break } }
            while ((FreeGB) -lt $j.need_gb -and (Get-Date) -lt $END -and -not (Paused)) { Say "$($j.name): waiting for memory (free $(FreeGB) GB, need $($j.need_gb))"; Start-Sleep -Seconds 120 }
            if ((Get-Date) -gt $END -or (Paused)) { break }
            foreach ($k in $j.env.Keys) { Set-Item -Path ("Env:" + $k) -Value $j.env[$k] }
            $exe = if ($j.kind -eq "retro") { $RPY } else { $PY }
            Say "$($j.name): start (try $try)"; $ran = $true
            $p = Start-Process -FilePath $exe -ArgumentList (@("-u","-W","ignore") + $j.args) -PassThru -NoNewWindow -RedirectStandardOutput "logs\q_$($Lane)_$($j.name)_$try.out" -RedirectStandardError "logs\q_$($Lane)_$($j.name)_$try.err"
            $null = $p.Handle; try { $p.PriorityClass = "Idle" } catch {}; try { $p.ProcessorAffinity = [IntPtr]$(if ($Lane -eq "A") { 0x0F00 } else { 0xF000 }) } catch {}
            $susp = $false; $hi = 0; $lo = 0; $killedForMem = $false; $ml = 0
            while (-not $p.HasExited) {
                Start-Sleep -Seconds 20
                Limit-Tree $p
                if ((Get-Date) -gt $END) { if ($susp) { Suspend $p $false }; KillTree $p; Say "$($j.name): stopped at the end time"; break }
                if (Paused) { if ($susp) { Suspend $p $false }; KillTree $p; Say "$($j.name): paused by queue\PAUSE (progress kept)"; break }
                if ((FreeGB) -lt 0.35) { $ml++ } else { $ml = 0 }; if ($ml -ge 3) { if ($susp) { Suspend $p $false }; KillTree $p; $killedForMem = $true; Say "$($j.name): killed, free memory under 0.35 GB for 1 minute (progress kept; restarts when memory is back)"; break }
                $c = CpuLoad
                if (-not $susp) { if ($c -gt 92) { $hi++ } else { $hi = 0 }; if ($hi -ge 3) { Suspend $p $true; $susp = $true; $lo = 0; Say "$($j.name): CPU at $c %, suspended" } }
                else { if ($c -lt 70) { $lo++ } else { $lo = 0 }; if ($lo -ge 2) { Suspend $p $false; $susp = $false; $hi = 0; Say "$($j.name): CPU at $c %, resumed" } }
            }
            foreach ($k in $j.env.Keys) { Remove-Item -Path ("Env:" + $k) -ErrorAction SilentlyContinue }
            if ($killedForMem) { $try--; Start-Sleep -Seconds 180; continue }
            if ($p.HasExited) {
                $pass = if ($j.check) { try { [bool](& $j.check) } catch { $false } } else { $p.ExitCode -eq 0 }
                if ($pass) { $ok = $true; Set-Content "$doneDir\$($j.name)" (Get-Date -Format s); Say "$($j.name): done" }
                elseif (-not (Paused) -and (Get-Date) -lt $END) { Say "$($j.name): not complete after exit $($p.ExitCode) (try $try)"; Start-Sleep -Seconds 120 }
            }
        }
        if (-not $ok -and (Get-Date) -lt $END -and -not (Paused)) { Set-Content $fm (Get-Date -Format s); Say "$($j.name): not finished after $($j.tries) tries; retried again in 30 minutes; lane continues" }
    }
    if (-not $ran -and (Get-Date) -lt $END) { Start-Sleep -Seconds 600 }
}
foreach ($c in $at_end) { Say "at end: $($c -join ' ')"; $p = Start-Process -FilePath $PY -ArgumentList (@("-u","-W","ignore") + $c) -PassThru -NoNewWindow -RedirectStandardOutput ("logs\q_" + $Lane + "_end.out") -RedirectStandardError ("logs\q_" + $Lane + "_end.err"); try { $p.PriorityClass = "Idle" } catch {}; $p.WaitForExit() }
Set-Content "queue\end_done_$Lane" (Get-Date -Format s); Say "lane $Lane finished"







