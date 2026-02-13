param(
  [switch]$HealthCheck,
  [string]$EnvFile = ".env",
  [int]$Port = 8000
)

 $ErrorActionPreference = 'Stop'

  function Write-Info($msg) {
    Write-Host "[INFO] $msg"
  }

  function Write-Err($msg) {
    Write-Host "[ERROR] $msg" -ForegroundColor Red
  }

  function Write-Warn($msg) {
    Write-Host "[WARN] $msg" -ForegroundColor Yellow
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

  function Import-EnvFile($path) {
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
      if (-not $k) { continue }

      $v = $parts[1].Trim()
      if ($v.StartsWith('"') -and $v.EndsWith('"') -and $v.Length -ge 2) {
        $v = $v.Substring(1, $v.Length - 2)
      } elseif ($v.StartsWith("'") -and $v.EndsWith("'") -and $v.Length -ge 2) {
        $v = $v.Substring(1, $v.Length - 2)
      }
      Set-Item -Path "Env:$k" -Value $v
    }
  }

  function Is-Truthy($value) {
    if ($null -eq $value) { return $false }
    $v = $value.ToString().Trim().ToLower()
    return $v -in @('1', 'true', 'yes')
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

    if ([System.IO.Path]::IsPathRooted($EnvFile)) {
      $envFile = $EnvFile
    } else {
      $envFile = Join-Path $PSScriptRoot $EnvFile
    }
    Import-EnvFile -path $envFile
    $token = Get-EnvValueFromFile -path $envFile -key 'TELEGRAM_BOT_TOKEN'

    if (-not $token) {
      throw "TELEGRAM_BOT_TOKEN not found in $envFile"
    }

    $tokenLen = $token.Length
    $last4 = if ($tokenLen -ge 4) { $token.Substring($tokenLen - 4, 4) } else { $token }
    Write-Info "Token loaded (length: $tokenLen, last4: $last4)"

    $enableNgrok = if ($env:ENABLE_NGROK -ne $null) { $env:ENABLE_NGROK } else { '1' }
    $enableWebhook = if ($env:ENABLE_WEBHOOK -ne $null) { $env:ENABLE_WEBHOOK } else { '1' }
    $enableLongPolling = if ($env:TELEGRAM_LONG_POLLING -ne $null) { $env:TELEGRAM_LONG_POLLING } else { '0' }
    if ($enableLongPolling -ne '0') {
      Write-Info 'TELEGRAM_LONG_POLLING=1, using getUpdates (no ngrok/webhook).'
      Write-Info 'Polling forwards updates to TELEGRAM_LOCAL_WEBHOOK_URL (default: http://127.0.0.1:8000/telegram).'
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

    if ($HealthCheck) {
      Write-Info 'Running health check (Google Vision + Dropbox)...'
      $healthScript = Join-Path $PSScriptRoot 'healthcheck.py'
      if (-not (Test-Path -Path $healthScript)) {
        throw "healthcheck.py not found: $healthScript"
      }
      & $pythonExe $healthScript
      if ($LASTEXITCODE -ne 0) {
        throw "Health check failed with exit code $LASTEXITCODE"
      }
      Write-Info 'Health check passed.'
      exit 0
    }

    $effectiveProvider = if ($env:AI_SUMMARY_PROVIDER) { $env:AI_SUMMARY_PROVIDER.ToLower().Trim() } else { '' }
    $needsOllama = (Is-Truthy $env:AI_SUMMARY_ENABLED) -and ($effectiveProvider -in @('ollama', 'local'))
    if ($needsOllama) {
      Ensure-Command 'ollama'
      $base = if ($env:OLLAMA_BASE_URL) { $env:OLLAMA_BASE_URL.TrimEnd('/') } else { 'http://127.0.0.1:11434' }
      $tagsUrl = "$base/api/tags"
      $ollamaReady = $false
      try {
        $null = Invoke-RestMethod -Method Get -Uri $tagsUrl -TimeoutSec 2
        $ollamaReady = $true
        Write-Info 'Ollama service already running.'
      } catch {
        Write-Info 'Starting Ollama service...'
        Start-Process -FilePath 'ollama' -ArgumentList 'serve'
      }

      if (-not $ollamaReady) {
        $deadlineOllama = (Get-Date).AddSeconds(20)
        while ((Get-Date) -lt $deadlineOllama) {
          try {
            $null = Invoke-RestMethod -Method Get -Uri $tagsUrl -TimeoutSec 2
            $ollamaReady = $true
            break
          } catch {
            Start-Sleep -Seconds 1
          }
        }
      }

      if (-not $ollamaReady) {
        throw "Ollama service did not become ready at $tagsUrl"
      }
      $modelName = if ($env:OLLAMA_MODEL) { $env:OLLAMA_MODEL.Trim() } else { 'qwen2.5:7b' }
      if ($modelName) {
        $tagsResp = Invoke-RestMethod -Method Get -Uri $tagsUrl -TimeoutSec 5
        $modelExists = $false
        if ($tagsResp -and $tagsResp.models) {
          foreach ($m in $tagsResp.models) {
            $tagName = ''
            if ($m.name) {
              $tagName = $m.name
            } elseif ($m.model) {
              $tagName = $m.model
            }
            if ($tagName -eq $modelName) {
              $modelExists = $true
              break
            }
          }
        }
        if (-not $modelExists) {
          Write-Info "Ollama model not found locally, pulling: $modelName"
          & ollama pull $modelName
          if ($LASTEXITCODE -ne 0) {
            throw "Failed to pull Ollama model: $modelName (exit code $LASTEXITCODE)"
          }
          Write-Info "Ollama model pull completed: $modelName"
        }
      }
      Write-Info "Ollama ready: $base (model: $modelName)"
    }

    $fallbackToLongPolling = $false

    if ($enableNgrok -ne '0') {
      Ensure-Command 'ngrok'
      Write-Info 'Starting ngrok...'
      Start-Process -FilePath 'ngrok' -ArgumentList "http $Port"

      if ($enableWebhook -ne '0') {
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
          Write-Warn 'Timeout waiting for ngrok public URL. Falling back to long polling.'
          $fallbackToLongPolling = $true
        } else {
          Write-Info "ngrok URL: $publicUrl"
          $webhookUrl = "$publicUrl/telegram"
          try {
            Write-Info 'Setting Telegram webhook...'
            $setResp = Invoke-RestMethod -Method Post -Uri "https://api.telegram.org/bot$token/setWebhook" -Body @{ url = $webhookUrl } -TimeoutSec 10
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
            Write-Warn "Webhook setup failed: $($_.Exception.Message)"
            Write-Warn 'Falling back to long polling.'
            $fallbackToLongPolling = $true
          }
        }
      } else {
        Write-Info 'ENABLE_WEBHOOK=0, skipping webhook setup.'
      }
    } else {
      Write-Info 'ENABLE_NGROK=0, skipping ngrok and webhook.'
    }

    if ($fallbackToLongPolling) {
      $env:TELEGRAM_LONG_POLLING = '1'
      Write-Info 'TELEGRAM_LONG_POLLING forced to 1 due to fallback.'
    }

    Write-Info 'Starting uvicorn...'
    Write-Info "Using env file: $envFile"
    Write-Info "Starting on port: $Port"
    Start-Process -FilePath $pythonExe -WorkingDirectory $PSScriptRoot -ArgumentList '-m','uvicorn','app:app','--host','0.0.0.0','--port',"$Port"
  } catch {
    Write-Err $_.Exception.Message
    exit 1
  }
