#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Stops and removes the elia_forecaster and rte_forecaster Windows services.
#>

$Nssm     = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\NSSM.NSSM_Microsoft.Winget.Source_8wekyb3d8bbwe\nssm-2.24-101-g897c7ad\win64\nssm.exe"
$services = @("EliaImbalanceForecaster", "RteImbalanceForecaster")

foreach ($name in $services) {
    $svc = Get-Service -Name $name -ErrorAction SilentlyContinue
    if ($svc) {
        Write-Host "Removing $name (status: $($svc.Status))..."
        if ($svc.Status -eq 'Running') {
            & $Nssm stop $name confirm | Out-Null
            Start-Sleep -Seconds 2
        }
        & $Nssm remove $name confirm | Out-Null
        Write-Host "  Removed." -ForegroundColor Green
    } else {
        Write-Host "$name not found — skipping." -ForegroundColor Yellow
    }
}
