# usage: powershell -File bg.ps1 <logname> <priority> <python args...>   (runs bbbenv python from the project root; output in logs\<logname>.out/.err)
param([string]$Name, [string]$Prio, [Parameter(ValueFromRemainingArguments=$true)]$Rest)
$wd = Split-Path -Parent $MyInvocation.MyCommand.Path
$p = Start-Process -FilePath "python" -WorkingDirectory $wd -ArgumentList (@("-u","-W","ignore") + $Rest) -PassThru -NoNewWindow -RedirectStandardOutput "$wd\logs\$Name.out" -RedirectStandardError "$wd\logs\$Name.err"
try { $p.PriorityClass = $Prio } catch {}
Write-Output "started $Name pid $($p.Id)"
