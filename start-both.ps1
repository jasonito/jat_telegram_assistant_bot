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

Write-Host "[INFO] Starting main bot with $MainEnv on port $MainPort"
$mainParams = @{
  EnvFile = $MainEnv
  Port = $MainPort
  SkipDepsInstall = $true
  ShowWindow = $effectiveShowShell
}
if ($EnableLogs -and $mainLog) {
  $mainParams["LogFile"] = $mainLog
}
& "$PSScriptRoot\start.ps1" @mainParams

Write-Host "[INFO] Starting chitchat bot with $ChitchatEnv on port $ChitchatPort"
$chitchatParams = @{
  EnvFile = $ChitchatEnv
  Port = $ChitchatPort
  SkipDepsInstall = $true
  ShowWindow = $effectiveShowShell
}
if ($EnableLogs -and $chitchatLog) {
  $chitchatParams["LogFile"] = $chitchatLog
}
& "$PSScriptRoot\start.ps1" @chitchatParams

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
