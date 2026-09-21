[CmdletBinding()]
param(
    [string]$Model = "models/Qwen3-4B-npu-4k",
    [string]$Revision = "d6fb03663d78ae5034d4594bfe9d92b35a5e213a",
    [Alias("Input")]
    [string]$InputFile = "examples/decisions.jsonl",
    [string]$Output,
    [int]$MaxTokens = 4096
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot ".")).Path
Push-Location $repo
$exitCode = 1
$sdkRoot = 'C:\Program Files\RyzenAI\1.8.0'
$previousSdkRoot = [Environment]::GetEnvironmentVariable('RYZEN_AI_INSTALLATION_PATH', 'Process')

try {
    if (-not (Test-Path -LiteralPath $sdkRoot -PathType Container)) {
        throw "Ryzen AI 1.8 SDK directory not found: $sdkRoot"
    }
    $env:RYZEN_AI_INSTALLATION_PATH = $sdkRoot
    $python = Join-Path $repo ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        throw "Project interpreter not found: $python. Create .venv and install the project first."
    }
    if ($MaxTokens -lt 1 -or $MaxTokens -gt 16384) {
        throw "MaxTokens must be between 1 and 16384; the selected model's context limit is also enforced."
    }
    if (-not (Test-Path -LiteralPath $Model -PathType Container)) {
        throw "Local NPU model directory not found: $Model"
    }
    if (-not (Test-Path -LiteralPath $InputFile -PathType Leaf)) {
        throw "Input JSONL file not found: $InputFile"
    }
    if (-not $Output) {
        $stamp = Get-Date -Format "yyyyMMdd-HHmmssfff"
        $Output = Join-Path $repo ("cache\results\semif-npu-{0}-{1}.jsonl" -f $stamp, $PID)
    }
    $outputPath = [IO.Path]::GetFullPath($Output)
    if (Test-Path -LiteralPath $outputPath) {
        throw "Refusing to overwrite existing output: $outputPath"
    }

    & $python -m semif_phase1.cli `
        --backend ryzenai-npu `
        --mode direct `
        --model $Model `
        --revision $Revision `
        --input $InputFile `
        --output $outputPath `
        --max-tokens $MaxTokens
    $exitCode = $LASTEXITCODE
    if ($exitCode -eq 0) {
        Write-Host "Wrote $outputPath"
    }
} finally {
    if ($null -eq $previousSdkRoot) {
        Remove-Item Env:RYZEN_AI_INSTALLATION_PATH -ErrorAction SilentlyContinue
    } else {
        $env:RYZEN_AI_INSTALLATION_PATH = $previousSdkRoot
    }
    Pop-Location
}
exit $exitCode
