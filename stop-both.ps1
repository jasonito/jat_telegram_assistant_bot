param(
  [int[]]$Ports = @(8000, 8001),
  [string[]]$AppTargets = @("app_main:app", "app_chitchat:app")
)

$ErrorActionPreference = "Stop"

function Match-Any([string]$text, [string[]]$needles) {
  foreach ($n in $needles) {
    if (-not [string]::IsNullOrWhiteSpace($n) -and $text -like "*$n*") {
      return $true
    }
  }
  return $false
}

$candidates = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -and $_.CommandLine -like "*-m uvicorn*" }

$toStop = @()
foreach ($p in $candidates) {
  $cmd = $p.CommandLine
  $matchTarget = Match-Any $cmd $AppTargets

  $matchPort = $false
  foreach ($port in $Ports) {
    if ($cmd -like "*--port $port*") {
      $matchPort = $true
      break
    }
  }

  if ($matchTarget -or $matchPort) {
    $toStop += $p
  }
}

if (-not $toStop -or $toStop.Count -eq 0) {
  Write-Host "[INFO] No matching bot process found."
  exit 0
}

$ids = @($toStop | ForEach-Object { $_.ProcessId })
Write-Host "[INFO] Stopping bot process IDs: $($ids -join ', ')"

foreach ($proc in $toStop) {
  try {
    Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop
    Write-Host "[OK] Stopped PID $($proc.ProcessId)"
  } catch {
    Write-Host "[WARN] Failed to stop PID $($proc.ProcessId): $($_.Exception.Message)"
  }
}
