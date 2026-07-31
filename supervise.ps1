<#
  supervise.ps1 — External keep-alive / auto-restart supervisor for the Jat bots.

  WHY THIS EXISTS
  ---------------
  start.ps1 / start-both.ps1 only *launch* uvicorn and then exit. Nothing watches
  the process afterwards. On 2026-07-28 the host lost network (DNS getaddrinfo
  failures, most likely a laptop sleep), the Telegram polling loop wedged, the
  in-process watchdog only logged "stale loop" without restarting, and news
  fetching produced nothing until a manual restart ~3 days later.

  This supervisor closes that gap WITHOUT touching app.py:
    * It polls each bot's /healthz endpoint.
    * When a bot is unhealthy AND the network is reachable, it restarts that bot
      by killing the port owner and re-invoking start.ps1 (same launch path).
    * When the network itself is down, it does NOTHING (waits) — restarting would
      not help and would only thrash. As soon as the network returns and the app
      is still wedged, it restarts within one grace window.

  LIMITATION: while the machine is asleep/hibernating, THIS script is frozen too.
  If the laptop keeps sleeping, also disable sleep (see README note printed by
  -Install). This supervisor's job is: recover fast once the machine is awake,
  and relaunch after a crash / reboot+login.

  USAGE
  -----
    # Run in the foreground (Ctrl+C to stop):
    powershell -ExecutionPolicy Bypass -File .\supervise.ps1

    # Register as a Scheduled Task that starts at logon and keeps running:
    powershell -ExecutionPolicy Bypass -File .\supervise.ps1 -Install

    # Remove the Scheduled Task:
    powershell -ExecutionPolicy Bypass -File .\supervise.ps1 -Uninstall
#>

param(
  [switch]$Install,
  [switch]$Uninstall,
  [string]$TaskName = "JatBotSupervisor",

  # How often to probe health.
  [int]$CheckIntervalSeconds = 30,
  # A bot must stay unhealthy this long (network up) before we restart it.
  [int]$UnhealthyGraceSeconds = 120,
  # Minimum spacing between restarts of the same bot.
  [int]$RestartCooldownSeconds = 120,
  # How long start.ps1 is allowed to wait for health during a restart.
  [int]$HealthWaitTimeoutSeconds = 90,

  # Network reachability probe (also implicitly validates DNS).
  [string]$NetworkProbeHost = "api.telegram.org",
  [int]$NetworkProbePort = 443,

  # Bots to supervise, as "EnvFile:Port" pairs (mirrors start-both.ps1).
  [string[]]$Targets = @(".env.main:8000", ".env.chitchat:8001")
)

$ErrorActionPreference = "Stop"
$ScriptDir = $PSScriptRoot
$ScriptPath = $MyInvocation.MyCommand.Path

function Get-LogPath {
  $dir = Join-Path $ScriptDir ".logs\runtime"
  if (-not (Test-Path -Path $dir)) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
  }
  return Join-Path $dir ("supervise-{0}.log" -f (Get-Date -Format 'yyyyMMdd'))
}

function Write-Log([string]$level, [string]$msg) {
  $line = "[{0}] [{1}] {2}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $level, $msg
  Write-Host $line
  try { Add-Content -Path (Get-LogPath) -Value $line -Encoding UTF8 } catch {}
}

# --- Network reachability (TCP connect with timeout; fails fast on DNS failure) ---
function Test-NetworkUp {
  $client = $null
  try {
    $client = New-Object System.Net.Sockets.TcpClient
    $iar = $client.BeginConnect($NetworkProbeHost, $NetworkProbePort, $null, $null)
    $ok = $iar.AsyncWaitHandle.WaitOne(3000, $false)
    if ($ok -and $client.Connected) {
      $client.EndConnect($iar)
      return $true
    }
    return $false
  } catch {
    return $false
  } finally {
    if ($client) { try { $client.Close() } catch {} }
  }
}

# --- PIDs listening on a TCP port (borrowed from start-both.ps1) ---
function Get-ListeningPidsForPort([int]$TargetPort) {
  $pids = @()
  try {
    $lines = netstat -ano -p tcp | Select-String -Pattern "^\s*TCP\s+\S+:$TargetPort\s+\S+\s+LISTENING\s+(\d+)\s*$"
    foreach ($line in $lines) {
      $text = $line.ToString()
      if ($text -match "LISTENING\s+(\d+)\s*$") {
        $pids += [int]$matches[1]
      }
    }
  } catch {}
  return @($pids | Select-Object -Unique)
}

# --- Health probe. Healthy = ok:true, and for long_polling also thread_alive && !stale ---
function Test-TargetHealthy([int]$port) {
  try {
    $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$port/healthz" -Method Get -TimeoutSec 4
  } catch {
    return $false
  }
  if ($null -eq $resp) { return $false }
  if (-not $resp.ok) { return $false }
  if ($resp.telegram_mode -eq 'long_polling') {
    if (-not $resp.telegram_poll) { return $false }
    if (-not $resp.telegram_poll.thread_alive) { return $false }
    if ($resp.telegram_poll.stale) { return $false }
  }
  return $true
}

# --- Kill the (python/uvicorn) process holding a port, so start.ps1 can rebind ---
function Stop-PortOwner([int]$port) {
  $pids = Get-ListeningPidsForPort -TargetPort $port
  foreach ($processId in $pids) {
    try {
      $proc = Get-Process -Id $processId -ErrorAction Stop
      $name = $proc.ProcessName
      # Guard: only stop python-hosted uvicorn, never an unrelated port owner.
      if ($name -notmatch 'python') {
        Write-Log "WARN" "Port $port owned by non-python process '$name' (pid=$processId); not killing."
        continue
      }
      Write-Log "INFO" "Stopping wedged process pid=$processId ($name) on port $port."
      Stop-Process -Id $processId -Force -ErrorAction Stop
    } catch {
      Write-Log "WARN" "Failed to stop pid=$processId on port ${port}: $($_.Exception.Message)"
    }
  }
  if ($pids.Count -gt 0) { Start-Sleep -Seconds 2 }
}

# --- Restart one bot via the normal launch path (start.ps1) ---
function Restart-Target([string]$envFile, [int]$port) {
  Write-Log "INFO" "Restarting bot env=$envFile port=$port ..."
  Stop-PortOwner -port $port
  try {
    & "$ScriptDir\start.ps1" -EnvFile $envFile -Port $port -SkipDepsInstall -HealthWaitTimeoutSeconds $HealthWaitTimeoutSeconds
    Write-Log "INFO" "Restart submitted for env=$envFile port=$port."
    return $true
  } catch {
    # start.ps1 throws if health did not become ready in time. The uvicorn
    # process was still launched (detached), so it usually recovers once the
    # network stabilizes; we simply re-check on the next cycle.
    Write-Log "WARN" "start.ps1 for env=$envFile port=$port returned an error: $($_.Exception.Message)"
    return $false
  }
}

# --- Parse "EnvFile:Port" into an object ---
function Parse-Target([string]$spec) {
  $idx = $spec.LastIndexOf(':')
  if ($idx -lt 1) { throw "Invalid target '$spec'. Expected 'EnvFile:Port'." }
  $envFile = $spec.Substring(0, $idx)
  $port = [int]$spec.Substring($idx + 1)
  return [pscustomobject]@{ EnvFile = $envFile; Port = $port }
}

# ---------------- Scheduled Task install / uninstall ----------------
function Install-Task {
  $psExe = (Get-Command powershell.exe).Source
  $arg = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$ScriptPath`""
  $action = New-ScheduledTaskAction -Execute $psExe -Argument $arg -WorkingDirectory $ScriptDir
  $trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
  $settings = New-ScheduledTaskSettingsSet `
      -AllowStartIfOnBatteries `
      -DontStopIfGoingOnBatteries `
      -StartWhenAvailable `
      -RestartCount 999 `
      -RestartInterval (New-TimeSpan -Minutes 1) `
      -ExecutionTimeLimit ([TimeSpan]::Zero) `
      -MultipleInstances IgnoreNew
  Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
      -Settings $settings -RunLevel Limited -Force | Out-Null
  Write-Log "INFO" "Registered Scheduled Task '$TaskName' (runs at logon of $env:USERNAME)."
  Write-Host ""
  Write-Host "[NEXT] The supervisor will auto-start at your next logon. To start it now without logging out:"
  Write-Host "         Start-ScheduledTask -TaskName '$TaskName'"
  Write-Host ""
  Write-Host "[NOTE] This supervisor is frozen while the machine SLEEPS. The 3-day gap looked like"
  Write-Host "       sleep/hibernate. To also prevent that (recommended for an always-on bot):"
  Write-Host "         powercfg /change standby-timeout-ac 0"
  Write-Host "         powercfg /change hibernate-timeout-ac 0"
  Write-Host "       (…-dc variants for battery). Adjust to taste."
}

function Uninstall-Task {
  try {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction Stop
    Write-Log "INFO" "Removed Scheduled Task '$TaskName'."
  } catch {
    Write-Log "WARN" "Could not remove task '$TaskName': $($_.Exception.Message)"
  }
}

if ($Install) { Install-Task; return }
if ($Uninstall) { Uninstall-Task; return }

# ---------------- Supervision loop ----------------
$targetObjs = @($Targets | ForEach-Object { Parse-Target $_ })
$state = @{}
foreach ($t in $targetObjs) {
  $state[$t.Port] = [pscustomobject]@{
    UnhealthySince = $null
    LastRestart    = [DateTime]::MinValue
    LastNetWarn    = [DateTime]::MinValue
  }
}

Write-Log "INFO" ("Supervisor started. Targets: {0}. interval=${CheckIntervalSeconds}s grace=${UnhealthyGraceSeconds}s cooldown=${RestartCooldownSeconds}s" -f ($Targets -join ", "))

while ($true) {
  $netUp = Test-NetworkUp
  $now = Get-Date

  foreach ($t in $targetObjs) {
    $st = $state[$t.Port]
    $healthy = Test-TargetHealthy -port $t.Port

    if ($healthy) {
      if ($st.UnhealthySince -ne $null) {
        Write-Log "INFO" "Bot port=$($t.Port) recovered (healthy again)."
      }
      $st.UnhealthySince = $null
      continue
    }

    # Unhealthy from here.
    if ($st.UnhealthySince -eq $null) {
      $st.UnhealthySince = $now
      Write-Log "WARN" "Bot port=$($t.Port) unhealthy (first detection)."
    }

    if (-not $netUp) {
      # Do not restart during a network outage — it cannot help and would thrash.
      if (($now - $st.LastNetWarn).TotalSeconds -ge 300) {
        Write-Log "WARN" "Network down (cannot reach ${NetworkProbeHost}:${NetworkProbePort}); deferring restart of port=$($t.Port)."
        $st.LastNetWarn = $now
      }
      continue
    }

    $unhealthyFor = ($now - $st.UnhealthySince).TotalSeconds
    $sinceRestart = ($now - $st.LastRestart).TotalSeconds
    if ($unhealthyFor -ge $UnhealthyGraceSeconds -and $sinceRestart -ge $RestartCooldownSeconds) {
      Write-Log "WARN" ("Bot port=$($t.Port) unhealthy for {0:N0}s with network UP; restarting." -f $unhealthyFor)
      [void](Restart-Target -envFile $t.EnvFile -port $t.Port)
      $st.LastRestart = Get-Date
      # Give the fresh process time to come up before counting unhealthy again.
      $st.UnhealthySince = $null
    }
  }

  Start-Sleep -Seconds $CheckIntervalSeconds
}
