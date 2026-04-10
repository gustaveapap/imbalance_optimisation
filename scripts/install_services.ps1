#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Registers elia_forecaster and rte_forecaster as Windows services using NSSM.
    Run once as Administrator. Services will start automatically on boot.

.USAGE
    Right-click PowerShell -> "Run as administrator"
    cd C:\Users\gusta\imbalance_optimisation
    .\scripts\install_services.ps1
#>

$ErrorActionPreference = 'Stop'

# ── Paths ─────────────────────────────────────────────────────────────────────
$Root    = "C:\Users\gusta\imbalance_optimisation"
$Python  = "$Root\venv\Scripts\python.exe"
$Nssm    = "C:\Users\gusta\AppData\Local\Microsoft\WinGet\Packages\NSSM.NSSM_Microsoft.Winget.Source_8wekyb3d8bbwe\nssm-2.24-101-g897c7ad\win64\nssm.exe"
$LogRoot = "$Root\logs"

# ── Validate prerequisites ─────────────────────────────────────────────────────
if (-not (Test-Path $Python)) {
    Write-Host "ERROR: Python not found at $Python"
    exit 1
}
if (-not (Test-Path $Nssm)) {
    Write-Host "ERROR: NSSM not found at $Nssm"
    exit 1
}

# ── Service definitions ────────────────────────────────────────────────────────
$eliaName        = "EliaImbalanceForecaster"
$eliaDisplay     = "Elia Imbalance Forecaster"
$eliaDescription = "Real-time Belgian grid (Elia) imbalance forecaster. Flask API on port 8080."
$eliaAppDir      = "$Root\forecasters\elia_forecaster"
$eliaScript      = "app.py"
$eliaLogDir      = "$LogRoot\elia_forecaster"

$rteName        = "RteImbalanceForecaster"
$rteDisplay     = "RTE Imbalance Forecaster"
$rteDescription = "French grid (RTE) imbalance forecast scheduler. Runs every 15 minutes via APScheduler."
$rteAppDir      = "$Root\forecasters\rte_forecaster"
$rteScript      = "run_forecast_scheduler.py"
$rteLogDir      = "$LogRoot\rte_forecaster"

# ── Helper function ────────────────────────────────────────────────────────────
function Install-ForecastService($name, $displayName, $description, $appDir, $script, $logDir) {

    Write-Host ""
    Write-Host "── $name ──────────────────────────────────"

    # Remove existing service if present
    $existing = Get-Service -Name $name -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "  Removing existing service..."
        if ($existing.Status -eq "Running") {
            & $Nssm stop $name confirm
            Start-Sleep -Seconds 2
        }
        & $Nssm remove $name confirm
        Start-Sleep -Seconds 1
    }

    # Ensure log directory exists
    if (-not (Test-Path $logDir)) {
        New-Item -ItemType Directory -Path $logDir | Out-Null
    }

    # Install service
    Write-Host "  Installing service..."
    & $Nssm install $name $Python $script

    # Core settings
    & $Nssm set $name AppDirectory $appDir
    & $Nssm set $name DisplayName $displayName
    & $Nssm set $name Description $description

    # UTF-8 encoding to avoid UnicodeEncodeError on Windows terminals
    & $Nssm set $name AppEnvironmentExtra "PYTHONIOENCODING=utf-8"

    # Stdout / stderr log files
    $stdoutLog = "$logDir\stdout.log"
    $stderrLog = "$logDir\stderr.log"
    & $Nssm set $name AppStdout $stdoutLog
    & $Nssm set $name AppStderr $stderrLog
    & $Nssm set $name AppStdoutCreationDisposition 4
    & $Nssm set $name AppStderrCreationDisposition 4

    # Log rotation: daily or 10 MB
    & $Nssm set $name AppRotateFiles 1
    & $Nssm set $name AppRotateSeconds 86400
    & $Nssm set $name AppRotateBytes 10485760

    # Restart on failure after 5 seconds
    & $Nssm set $name AppExit Default Restart
    & $Nssm set $name AppRestartDelay 5000

    # Auto-start on boot
    & $Nssm set $name Start SERVICE_AUTO_START

    Write-Host "  OK -- $name configured."
}

# ── Install both services ──────────────────────────────────────────────────────
Install-ForecastService $eliaName $eliaDisplay $eliaDescription $eliaAppDir $eliaScript $eliaLogDir
Install-ForecastService $rteName  $rteDisplay  $rteDescription  $rteAppDir  $rteScript  $rteLogDir

# ── Start both services ────────────────────────────────────────────────────────
Write-Host ""
Write-Host "── Starting services ────────────────────────"

& $Nssm start $eliaName
Start-Sleep -Seconds 2
$eliaStatus = (Get-Service -Name $eliaName).Status
Write-Host "  $eliaName status: $eliaStatus"

& $Nssm start $rteName
Start-Sleep -Seconds 2
$rteStatus = (Get-Service -Name $rteName).Status
Write-Host "  $rteName status: $rteStatus"

# ── Summary ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Services installed and started."
Write-Host "Manage via: services.msc or sc.exe"
Write-Host "Logs:       $LogRoot"
Write-Host "Elia UI:    http://localhost:8080"
