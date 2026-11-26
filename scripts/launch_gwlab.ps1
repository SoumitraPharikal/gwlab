# Windows PowerShell launcher for GWLab (FastAPI + Next.js)
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$LauncherArgs
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $RepoRoot
try {
    python scripts\launch_gwlab.py @LauncherArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
