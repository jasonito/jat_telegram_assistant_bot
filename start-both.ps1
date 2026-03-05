param(
  [string]$MainEnv = ".env.main",
  [int]$MainPort = 8000,
  [string]$ChitchatEnv = ".env.chitchat",
  [int]$ChitchatPort = 8001,
  [switch]$ShowShell,
  [switch]$EnableLogs,
  [string]$LogDir = ".\logs",
  [switch]$Monitor
)

$ErrorActionPreference = "Stop"

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

function Get-NextFreePort([int]$StartPort, [int[]]$ReservedPorts) {
  $scanLimit = [Math]::Min(65535, $StartPort + 50)
  for ($candidate = $StartPort; $candidate -le $scanLimit; $candidate++) {
    if ($ReservedPorts -contains $candidate) {
      continue
    }
    $pids = Get-ListeningPidsForPort -TargetPort $candidate
    if ($pids.Count -eq 0) {
      return $candidate
    }
  }
  throw "No free port found in range $StartPort-$scanLimit."
}

function Ensure-SharedDependencies {
  $venvDir = Join-Path $PSScriptRoot '.venv'
  $venvPython = Join-Path $venvDir 'Scripts\python.exe'
  if (-not (Test-Path -Path $venvPython)) {
    Write-Host "[INFO] Virtualenv not found. Creating .venv..."
    & python -m venv $venvDir
  }

  if (-not (Test-Path -Path $venvPython)) {
    throw "Virtualenv creation failed. Expected: $venvPython"
  }

  $requirements = Join-Path $PSScriptRoot 'requirements.txt'
  if (-not (Test-Path -Path $requirements)) {
    throw "requirements.txt not found: $requirements"
  }

  $requirementsHash = (Get-FileHash -Path $requirements -Algorithm SHA256).Hash
  $requirementsStamp = Join-Path $venvDir 'requirements.sha256'
  $installedHash = if (Test-Path -Path $requirementsStamp) {
    (Get-Content -Path $requirementsStamp -Raw).Trim()
  } else {
    ''
  }

  if ($installedHash -eq $requirementsHash) {
    Write-Host "[INFO] Dependencies unchanged, shared install skipped."
    return
  }

  $constraints = Join-Path $PSScriptRoot 'constraints.txt'
  if (Test-Path -Path $constraints) {
    Write-Host "[INFO] Installing shared Python dependencies (with constraints)..."
    & $venvPython -m pip install -r $requirements -c $constraints
  } else {
    Write-Host "[INFO] Installing shared Python dependencies..."
    & $venvPython -m pip install -r $requirements
  }
  if ($LASTEXITCODE -ne 0) {
    throw "pip install failed with exit code $LASTEXITCODE"
  }
  Set-Content -Path $requirementsStamp -Value $requirementsHash -NoNewline
}

$mainLog = $null
$chitchatLog = $null
if ($EnableLogs -or $Monitor) {
  $EnableLogs = $true
  if (-not (Test-Path -Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
  }
  $mainLog = Join-Path $LogDir "main.log"
  $chitchatLog = Join-Path $LogDir "chitchat.log"
}

$effectiveShowShell = [bool]$ShowShell
if ($Monitor) {
  # Monitor mode streams logs in current shell; avoid opening extra blank child windows.
  $effectiveShowShell = $false
}

Ensure-SharedDependencies

$mainSelectedPort = Get-NextFreePort -StartPort $MainPort -ReservedPorts @()
if ($mainSelectedPort -ne $MainPort) {
  Write-Host "[WARN] Requested main port $MainPort is unavailable. Preselected fallback: $mainSelectedPort"
}

$chitchatSelectedPort = Get-NextFreePort -StartPort $ChitchatPort -ReservedPorts @($mainSelectedPort)
if ($chitchatSelectedPort -ne $ChitchatPort) {
  Write-Host "[WARN] Requested chitchat port $ChitchatPort is unavailable/conflicts. Preselected fallback: $chitchatSelectedPort"
}
if ($mainSelectedPort -eq $chitchatSelectedPort) {
  throw "Internal port selection conflict: both bots resolved to port $mainSelectedPort."
}

Write-Host "[INFO] Starting main bot with $MainEnv on port $mainSelectedPort"
$mainParams = @{
  EnvFile = $MainEnv
  Port = $mainSelectedPort
  SkipDepsInstall = $true
  ShowWindow = $effectiveShowShell
}
if ($EnableLogs -and $mainLog) {
  $mainParams["LogFile"] = $mainLog
}
& "$PSScriptRoot\start.ps1" @mainParams
if ($LASTEXITCODE -ne 0) {
  throw "Main bot start failed with exit code $LASTEXITCODE"
}

Write-Host "[INFO] Starting chitchat bot with $ChitchatEnv on port $chitchatSelectedPort"
$chitchatParams = @{
  EnvFile = $ChitchatEnv
  Port = $chitchatSelectedPort
  SkipDepsInstall = $true
  ShowWindow = $effectiveShowShell
}
if ($EnableLogs -and $chitchatLog) {
  $chitchatParams["LogFile"] = $chitchatLog
}
& "$PSScriptRoot\start.ps1" @chitchatParams
if ($LASTEXITCODE -ne 0) {
  throw "Chitchat bot start failed with exit code $LASTEXITCODE"
}

Write-Host "[INFO] Both start commands submitted."
if ($EnableLogs) {
  Write-Host "[INFO] Logs:"
  Write-Host "  - $mainLog.out.log"
  Write-Host "  - $mainLog.err.log"
  Write-Host "  - $chitchatLog.out.log"
  Write-Host "  - $chitchatLog.err.log"
  Write-Host "[INFO] Tail logs with: Get-Content $mainLog.out.log,$mainLog.err.log,$chitchatLog.out.log,$chitchatLog.err.log -Wait"
}
if ($Monitor) {
  Write-Host "[INFO] Monitoring logs (Ctrl+C to stop monitor, bots keep running)..."
  $watchFiles = @("$mainLog.out.log", "$mainLog.err.log", "$chitchatLog.out.log", "$chitchatLog.err.log")
  foreach ($wf in $watchFiles) {
    if (-not (Test-Path -Path $wf)) {
      New-Item -ItemType File -Path $wf -Force | Out-Null
    }
  }
  Get-Content $watchFiles -Wait
}
