param(
  [string]$TaskName = "JAT Telegram Assistant Bot Main",
  [string]$EnvFile = ".env.main",
  [int]$Port = 8000,
  [ValidateSet("Logon", "Startup")]
  [string]$Trigger = "Logon"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$startScript = Join-Path $repoRoot "start.ps1"

if (-not (Test-Path -Path $startScript)) {
  throw "start.ps1 not found: $startScript"
}

$envPath = if ([System.IO.Path]::IsPathRooted($EnvFile)) {
  $EnvFile
} else {
  Join-Path $repoRoot $EnvFile
}

if (-not (Test-Path -Path $envPath)) {
  throw "Env file not found: $envPath"
}

$powerShellExe = Join-Path $env:WINDIR "System32\WindowsPowerShell\v1.0\powershell.exe"
if (-not (Test-Path -Path $powerShellExe)) {
  throw "powershell.exe not found: $powerShellExe"
}

$escapedStartScript = $startScript.Replace('"', '""')
$escapedEnvPath = $envPath.Replace('"', '""')
$arguments = "-NoProfile -ExecutionPolicy Bypass -File ""$escapedStartScript"" -EnvFile ""$escapedEnvPath"" -Port $Port"

$action = New-ScheduledTaskAction -Execute $powerShellExe -Argument $arguments -WorkingDirectory $repoRoot

if ($Trigger -eq "Startup") {
  $taskTrigger = New-ScheduledTaskTrigger -AtStartup
  $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited
} else {
  $taskTrigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
  $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
}

$settings = New-ScheduledTaskSettingsSet `
  -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries `
  -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
  -MultipleInstances IgnoreNew `
  -StartWhenAvailable

$task = New-ScheduledTask -Action $action -Trigger $taskTrigger -Principal $principal -Settings $settings

Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force | Out-Null

Write-Host "Registered scheduled task:"
Write-Host "  Name: $TaskName"
Write-Host "  Trigger: $Trigger"
Write-Host "  EnvFile: $envPath"
Write-Host "  Port: $Port"
