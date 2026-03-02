param(
  [switch]$HealthCheck,
  [string]$EnvFile = ".env",
  [int]$Port = 8000,
  [switch]$ShowWindow,
  [string]$LogFile = "",
  [switch]$SkipDepsInstall
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

  function Get-NgrokPublicUrlForPort([int]$targetPort) {
    $resp = Invoke-RestMethod -Uri 'http://127.0.0.1:4040/api/tunnels' -Method Get -TimeoutSec 2
    if (-not $resp -or -not $resp.tunnels) {
      return $null
    }
    foreach ($t in $resp.tunnels) {
      $publicUrl = $t.public_url
      $proto = $t.proto
      $addr = ''
      if ($t.config -and $t.config.addr) {
        $addr = $t.config.addr.ToString().ToLower()
      }
      if (-not $publicUrl -or -not $proto) { continue }
      if ($proto -ne 'https') { continue }
      if (-not $addr) { continue }
      if ($addr -match ":$targetPort$" -or $addr -match "localhost:$targetPort$" -or $addr -match "127\.0\.0\.1:$targetPort$") {
        return $publicUrl
      }
    }
    return $null
  }

  function Get-ListeningPidsForPort([int]$targetPort) {
    $pids = @()
    try {
      $lines = netstat -ano -p tcp | Select-String -Pattern "^\s*TCP\s+\S+:$targetPort\s+\S+\s+LISTENING\s+(\d+)\s*$"
      foreach ($line in $lines) {
        $text = $line.ToString()
        if ($text -match "LISTENING\s+(\d+)\s*$") {
          $pids += [int]$matches[1]
        }
      }
    } catch {}
    return @($pids | Select-Object -Unique)
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
    $appModule = Get-EnvValueFromFile -path $envFile -key 'APP_MODULE'
    if (-not $appModule) {
      $appModule = 'app'
    }

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

    if (-not $SkipDepsInstall) {
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
      $needsInstall = ($installedHash -ne $requirementsHash)

      if ($needsInstall) {
        $constraints = Join-Path $PSScriptRoot 'constraints.txt'
        if (Test-Path -Path $constraints) {
          Write-Info 'Installing Python dependencies (with constraints)...'
          & $pythonExe -m pip install -r $requirements -c $constraints
        } else {
          Write-Info 'Installing Python dependencies...'
          & $pythonExe -m pip install -r $requirements
        }
        if ($LASTEXITCODE -ne 0) {
          throw "pip install failed with exit code $LASTEXITCODE"
        }
        Set-Content -Path $requirementsStamp -Value $requirementsHash -NoNewline
      } else {
        Write-Info 'Dependencies unchanged, skipping pip install.'
      }
    } else {
      Write-Info 'Skipping dependency install (-SkipDepsInstall).'
    }

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
        if ($ShowWindow) {
          Start-Process -FilePath 'ollama' -ArgumentList 'serve'
        } else {
          Start-Process -FilePath 'ollama' -ArgumentList 'serve' -WindowStyle Hidden
        }
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
      if ($enableWebhook -ne '0') {
        $deadline = (Get-Date).AddSeconds(30)
        $publicUrl = $null

        try {
          $publicUrl = Get-NgrokPublicUrlForPort -targetPort $Port
          if ($publicUrl) {
            Write-Info "Reusing ngrok tunnel for port ${Port}: $publicUrl"
          }
        } catch {}

        if (-not $publicUrl) {
          Write-Info "Starting ngrok for port $Port..."
          try {
            if ($ShowWindow) {
              Start-Process -FilePath 'ngrok' -ArgumentList "http $Port"
            } else {
              Start-Process -FilePath 'ngrok' -ArgumentList "http $Port" -WindowStyle Hidden
            }
          } catch {
            Write-Warn "Failed to start ngrok for port ${Port}: $($_.Exception.Message)"
            Write-Warn 'Falling back to long polling.'
            $fallbackToLongPolling = $true
          }
        }

        if (-not $fallbackToLongPolling) {
          Write-Info "Waiting for ngrok public URL (port $Port)..."
          while ((Get-Date) -lt $deadline) {
            try {
              $publicUrl = Get-NgrokPublicUrlForPort -targetPort $Port
              if ($publicUrl) { break }
            } catch {
              Start-Sleep -Seconds 1
              continue
            }
            Start-Sleep -Seconds 1
          }
        }

        if (-not $fallbackToLongPolling -and -not $publicUrl) {
          Write-Warn "Timeout waiting for ngrok public URL on port $Port. Falling back to long polling."
          $fallbackToLongPolling = $true
        } elseif (-not $fallbackToLongPolling) {
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
    Write-Info "App module: $appModule"
    Write-Info "Starting on port: $Port"
    $occupiedPids = Get-ListeningPidsForPort -targetPort $Port
    if ($occupiedPids.Count -gt 0) {
      throw "Port $Port is already in use by PID(s): $($occupiedPids -join ', '). Stop old process first."
    }
    if ($appModule.Contains(':')) {
      $appTarget = $appModule
    } else {
      $appTarget = "$appModule`:app"
    }
    $uvicornArgs = @('-u','-m','uvicorn',$appTarget,'--host','0.0.0.0','--port',"$Port")
    $startParams = @{
      FilePath = $pythonExe
      WorkingDirectory = $PSScriptRoot
      ArgumentList = $uvicornArgs
    }
    if (-not $ShowWindow) {
      $startParams["WindowStyle"] = "Hidden"
    }
    if ($LogFile) {
      $logBase = if ([System.IO.Path]::IsPathRooted($LogFile)) { $LogFile } else { Join-Path $PSScriptRoot $LogFile }
      $logDir = Split-Path -Parent $logBase
      if ($logDir -and -not (Test-Path -Path $logDir)) {
        New-Item -ItemType Directory -Path $logDir -Force | Out-Null
      }
      $stdoutLog = "$logBase.out.log"
      $stderrLog = "$logBase.err.log"
      Write-Info "Redirecting uvicorn logs to:"
      Write-Info "  STDOUT: $stdoutLog"
      Write-Info "  STDERR: $stderrLog"
      $startParams["RedirectStandardOutput"] = $stdoutLog
      $startParams["RedirectStandardError"] = $stderrLog
    }
    Start-Process @startParams
  } catch {
    Write-Err $_.Exception.Message
    exit 1
  }
