param(
  [string]$MainEnv = ".env.main",
  [int]$MainPort = 8000,
  [string]$ChitchatEnv = ".env.chitchat",
  [int]$ChitchatPort = 8001
)

$ErrorActionPreference = "Stop"

Write-Host "[INFO] Starting main bot with $MainEnv on port $MainPort"
& "$PSScriptRoot\start.ps1" -EnvFile $MainEnv -Port $MainPort

Write-Host "[INFO] Starting chitchat bot with $ChitchatEnv on port $ChitchatPort"
& "$PSScriptRoot\start.ps1" -EnvFile $ChitchatEnv -Port $ChitchatPort

Write-Host "[INFO] Both start commands submitted."
