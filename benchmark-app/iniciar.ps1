# Abre o app de benchmark do Juiz de Bocha em http://localhost:8501
# Uso no PowerShell:  & "C:\Users\Jonathas\Downloads\ColabJB\benchmark-app\iniciar.ps1"
Set-Location $PSScriptRoot

$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) {
    $uv = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe"
}
if (-not (Test-Path $uv)) {
    Write-Host "Nao encontrei o uv. Instale com:  winget install --id astral-sh.uv -e"
    exit 1
}

& $uv run streamlit run app.py --server.port 8501 --server.headless false
