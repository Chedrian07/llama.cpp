param(
  [Parameter(Mandatory=$true)][string]$Model,
  [string]$Root = "C:\Users\-_-kc\Desktop\llama.cpp",
  [string]$BuildDir = "C:\Users\-_-kc\Desktop\llama.cpp\build-mcd-cuda",
  [string]$Out = "C:\Users\-_-kc\Desktop\llama.cpp\results\mcd\windows_llama_bench.json",
  [string]$Prompt = "512,2048",
  [int]$Gen = 64,
  [int]$Repetitions = 1,
  [int]$NGpuLayers = 99,
  [string]$Device = "CUDA0/CUDA1",
  [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
if (Get-Variable PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
  $PSNativeCommandUseErrorActionPreference = $false
}

[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
chcp 65001 | Out-Null

Set-Location $Root
$Bin = Join-Path $BuildDir "bin\Release\llama-bench.exe"
$OutDir = Split-Path -Parent $Out
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$OldErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $Bin `
  -m $Model `
  -p $Prompt `
  -n $Gen `
  -r $Repetitions `
  -o json `
  --no-warmup `
  -ngl $NGpuLayers `
  -dev $Device `
  @ExtraArgs *> $Out
$Code = $LASTEXITCODE
$ErrorActionPreference = $OldErrorActionPreference

exit $Code
