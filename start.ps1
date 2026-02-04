 $ErrorActionPreference = 'Stop'

  function Write-Info($msg) {
    Write-Host "[INFO] $msg"
  }

  function Write-Err($msg) {
    Write-Host "[ERROR] $msg" -ForegroundColor Red
  }

  function Get-EnvValueFromFile($path, $key) {
    if (-not (Test-Path -Path $path)) {
      throw "Env file not found: $path"
    }

    $lines = Get-Content -Path $path -ErrorAction Stop
    foreach ($line in $lines) {
      $trim = $line.Trim()
      if ($trim.Length -eq 0) { continue }
      if ($trim.StartsWith('#')) { continue }

      $parts = $trim -split '=', 2
      if ($parts.Count -lt 2) { continue }

      $k = $parts[0].Trim()
      if ($k -ne $key) { continue }

      $v = $parts[1].Trim()
      if ($v.StartsWith('"') -and $v.EndsWith('"') -and $v.Length -ge 2) {
        $v = $v.Substring(1, $v.Length - 2)
      } elseif ($v.StartsWith("'") -and $v.EndsWith("'") -and $v.Length -ge 2) {
        $v = $v.Substring(1, $v.Length - 2)
      }
      return $v.Trim()
    }

    return $null
  }

  function Ensure-Command($name) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if (-not $cmd) {
      throw "Command not found: $name"
    }
  }

  try {
    Ensure-Command 'python'

    $venvDir = Join-Path $PSScriptRoot '.venv'
    $venvPython = Join-Path $venvDir 'Scripts\python.exe'
    if (-not (Test-Path -Path $venvPython)) {
      Write-Info "Virtualenv not found. Creating .venv..."
      & python -m venv $venvDir
    }

    $venvPython = Join-Path $venvDir 'Scripts\python.exe'
    if (-not (Test-Path -Path $venvPython)) {
      throw "Virtualenv creation failed. Expected: $venvPython"
    }

    $envFile = Join-Path $PSScriptRoot '.env'
    $token = Get-EnvValueFromFile -path $envFile -key 'TELEGRAM_BOT_TOKEN'

    if (-not $token) {
      throw 'TELEGRAM_BOT_TOKEN not found in .env'
    }

    $tokenLen = $token.Length
    $last4 = if ($tokenLen -ge 4) { $token.Substring($tokenLen - 4, 4) } else { $token }
    Write-Info "Token loaded (length: $tokenLen, last4: $last4)"

    $enableNgrok = if ($env:ENABLE_NGROK -ne $null) { $env:ENABLE_NGROK } else { '1' }
    $enableWebhook = if ($env:ENABLE_WEBHOOK -ne $null) { $env:ENABLE_WEBHOOK } else { '1' }
    $enableLongPolling = if ($env:TELEGRAM_LONG_POLLING -ne $null) { $env:TELEGRAM_LONG_POLLING } else { '0' }
    if ($enableLongPolling -ne '0') {
      Write-Info 'TELEGRAM_LONG_POLLING=1, using getUpdates (no ngrok/webhook).'
      Write-Info 'Polling forwards updates to http://127.0.0.1:8000/telegram by default.'
      $enableNgrok = '0'
      $enableWebhook = '0'
    }

    $pythonExe = $venvPython
    Ensure-Command $pythonExe

    $requirements = Join-Path $PSScriptRoot 'requirements.txt'
    if (-not (Test-Path -Path $requirements)) {
      throw "requirements.txt not found: $requirements"
    }

    Write-Info 'Installing Python dependencies...'
    & $pythonExe -m pip install -r $requirements

    if ($enableNgrok -ne '0') {
      Ensure-Command 'ngrok'
    }

    Write-Info 'Starting uvicorn...'
    Start-Process -FilePath $pythonExe -WorkingDirectory $PSScriptRoot -ArgumentList '-m','uvicorn','app:app','--host','0.0.0.0','--port','8000'

    if ($enableNgrok -eq '0') {
      Write-Info 'ENABLE_NGROK=0, skipping ngrok and webhook.'
      exit 0
    }

    Write-Info 'Starting ngrok...'
    Start-Process -FilePath 'ngrok' -ArgumentList 'http 8000'

    if ($enableWebhook -eq '0') {
      Write-Info 'ENABLE_WEBHOOK=0, skipping webhook setup.'
      exit 0
    }

    $deadline = (Get-Date).AddSeconds(30)
    $publicUrl = $null

    Write-Info 'Waiting for ngrok public URL...'
    while ((Get-Date) -lt $deadline) {
      try {
        $resp = Invoke-RestMethod -Uri 'http://127.0.0.1:4040/api/tunnels' -Method Get -TimeoutSec 2
        foreach ($t in $resp.tunnels) {
          if ($t.public_url -and $t.public_url.StartsWith('https://')) {
            $publicUrl = $t.public_url
            break
          }
        }
        if ($publicUrl) { break }
      } catch {
        Start-Sleep -Seconds 1
        continue
      }
      Start-Sleep -Seconds 1
    }

    if (-not $publicUrl) {
      throw 'Timeout waiting for ngrok public URL (check ngrok and 4040).'
    }

    Write-Info "ngrok URL: $publicUrl"

    $webhookUrl = "$publicUrl/telegram"
    Write-Info 'Setting Telegram webhook...'
    $setResp = Invoke-RestMethod -Method Post -Uri "https://api.telegram.org/bot$token/setWebhook" -Body @{ url =
  $webhookUrl } -TimeoutSec 10
    if (-not $setResp.ok) {
      throw "Telegram setWebhook failed: $($setResp | ConvertTo-Json -Depth 10)"
    }

    Write-Info 'Verifying Telegram webhook...'
    $infoResp = Invoke-RestMethod -Method Get -Uri "https://api.telegram.org/bot$token/getWebhookInfo" -TimeoutSec 10
    if (-not $infoResp.ok) {
      throw "Telegram getWebhookInfo failed: $($infoResp | ConvertTo-Json -Depth 10)"
    }

    $resultUrl = $infoResp.result.url
    if ($resultUrl -ne $webhookUrl) {
      throw "Webhook mismatch. Expected: $webhookUrl, Got: $resultUrl"
    }

    Write-Info 'Webhook set and verified.'
  } catch {
    Write-Err $_.Exception.Message
    exit 1
  }
