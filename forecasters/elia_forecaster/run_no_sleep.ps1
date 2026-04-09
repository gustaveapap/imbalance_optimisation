# Empecher mise en veille avec API Windows
Add-Type @'
using System;
using System.Runtime.InteropServices;

public class PowerManagement {
    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
    
    public const uint ES_CONTINUOUS = 0x80000000;
    public const uint ES_SYSTEM_REQUIRED = 0x00000001;
    public const uint ES_DISPLAY_REQUIRED = 0x00000002;
}
'@

Write-Host "Desactivation de la mise en veille..." -ForegroundColor Green
[PowerManagement]::SetThreadExecutionState(
    [PowerManagement]::ES_CONTINUOUS -bor 
    [PowerManagement]::ES_SYSTEM_REQUIRED -bor 
    [PowerManagement]::ES_DISPLAY_REQUIRED
) | Out-Null

Write-Host "Lancement de Elia Forecaster..." -ForegroundColor Cyan
Write-Host "Pour arreter : Ctrl+C puis fermer cette fenetre" -ForegroundColor Yellow
Write-Host ""

python app.py

Write-Host ""
Write-Host "Reactivation de la mise en veille..." -ForegroundColor Green
[PowerManagement]::SetThreadExecutionState([PowerManagement]::ES_CONTINUOUS) | Out-Null