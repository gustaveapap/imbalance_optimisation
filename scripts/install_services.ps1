#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Registers elia_forecaster and rte_forecaster as Windows services using NSSM.
    Run once as Administrator. Services will start automatically on boot.

.USAGE
    Right-click PowerShell → "Run as administrator"
    cd C:\Users\gusta\imbalance_optimisation
    .\scripts\install_services.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ── Paths ────────────────────────────────────────────────────────────────────
$Root    = "C:\Users\gusta\imbalance_optimisation"
$Python  = "$Root\venv\Scripts\python.exe"
$Nssm    = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\NSSM.NSSM_Microsoft.Winget.Source_8wekyb3d8bbwe\nssm-2.24-101-g897c7ad\win64\nssm.exe"
$LogRoot = "$Root\logs"

# ── Validate prerequisites ────────────────────────────────────────────────────
foreach ($path in @($Python, $Nssm)) {
    if (-not (Test-Path $path)) {
        Write-Error "Not found: $path"
        exit 1
    }
}

# ── Service definitions ───────────────────────────────────────────────────────
$services = @(
    @{
        Name        = "EliaImbalanceForecaster"
        DisplayName = "Elia Imbalance Forecaster"
        Description = "Real-time Belgian grid (Elia) imbalance forecaster. Flask API on port 8080."
        AppDir      = "$Root\forecasters\elia_forecaster"
        Script      = "app.py"
        LogDir      = "$LogRoot\elia_forecaster"
    },
    @{
        Name        = "RteImbalanceForecaster"
        DisplayName = "RTE Imbalance Forecaster"
        Description = "French grid (RTE) imbalance forecast scheduler. Runs every 15 minutes via APScheduler."
        AppDir      = "$Root\forecasters\rte_forecaster"
        Script      = "run_forecast_scheduler.py"
        LogDir      = "$LogRoot\rte_forecaster"
    }
)

# ── Install / reconfigure each service ───────────────────────────────────────
foreach ($svc in $services) {
    $name = $svc.Name

    Write-Host "`n── $name ──────────────────────────────────" -ForegroundColor Cyan

    # Remove existing service cleanly if present
    $existing = Get-Service -Name $name -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "  Removing existing service..." -ForegroundColor Yellow
        if ($existing.Status -eq 'Running') {
            & $Nssm stop $name confirm | Out-Null
            Start-Sleep -Seconds 2
        }
        & $Nssm remove $name confirm | Out-Null
        Start-Sleep -Seconds 1
    }

    # Ensure log directory exists
    New-Item -ItemType Directory -Force -Path $svc.LogDir | Out-Null

    # Install
    Write-Host "  Installing service..."
    & $Nssm install $name $Python $svc.Script

    # Core settings
    & $Nssm set $name AppDirectory      $svc.AppDir
    & $Nssm set $name DisplayName       $svc.DisplayName
    & $Nssm set $name Description       $svc.Description

    # Encoding — avoids UnicodeEncodeError on Windows cp1252 terminals
    & $Nssm set $name AppEnvironmentExtra "PYTHONIOENCODING=utf-8"

    # Stdout / stderr → rotating log files
    & $Nssm set $name AppStdout                    "$($svc.LogDir)\stdout.log"
    & $Nssm set $name AppStderr                    "$($svc.LogDir)\stderr.log"
    & $Nssm set $name AppStdoutCreationDisposition 4   # append
    & $Nssm set $name AppStderrCreationDisposition 4   # append
    & $Nssm set $name AppRotateFiles               1
    & $Nssm set $name AppRotateSeconds             86400  # rotate daily
    & $Nssm set $name AppRotateBytes               10485760  # 10 MB max per file

    # Restart on failure — wait 5 s before restarting
    & $Nssm set $name AppExit         Default Restart
    & $Nssm set $name AppRestartDelay 5000

    # Startup type — automatic (starts on boot, before login)
    & $Nssm set $name Start SERVICE_AUTO_START

    Write-Host "  OK — $name configured." -ForegroundColor Green
}

# ── Start services ────────────────────────────────────────────────────────────
Write-Host "`n── Starting services ────────────────────────" -ForegroundColor Cyan
foreach ($svc in $services) {
    Write-Host "  Starting $($svc.Name)..."
    & $Nssm start $($svc.Name)
    Start-Sleep -Seconds 2
    $status = (Get-Service -Name $svc.Name).Status
    Write-Host "  Status: $status" -ForegroundColor $(if ($status -eq 'Running') { 'Green' } else { 'Yellow' })
}

# ── Summary ───────────────────────────────────────────────────────────────────
Write-Host "`n════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host " Services installed and started." -ForegroundColor Green
Write-Host " Manage via:  services.msc  or  sc.exe" -ForegroundColor White
Write-Host " Logs:        $LogRoot" -ForegroundColor White
Write-Host " Elia UI:     http://localhost:8080" -ForegroundColor White
Write-Host "════════════════════════════════════════════`n" -ForegroundColor Cyan
