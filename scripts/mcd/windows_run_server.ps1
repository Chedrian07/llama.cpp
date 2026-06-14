param(
  [Parameter(Mandatory=$true)][string]$Model,
  [string]$Root = "C:\Users\-_-kc\Desktop\llama.cpp",
  [string]$BuildDir = "C:\Users\-_-kc\Desktop\llama.cpp\build-mcd-cuda",
  [string]$HostAddress = "169.254.21.157",
  [int]$Port = 18080,
  [int]$Ctx = 32768,
  [string]$NGpuLayers = "all",
  [string]$SlotSavePath = "C:\Users\-_-kc\Desktop\llama.cpp\results\mcd\windows_slots",
  [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$Bin = Join-Path $BuildDir "bin\Release\llama-server.exe"
New-Item -ItemType Directory -Force -Path $SlotSavePath | Out-Null

& $Bin `
  --host $HostAddress `
  --port $Port `
  --model $Model `
  --ctx-size $Ctx `
  --n-gpu-layers $NGpuLayers `
  --slot-save-path $SlotSavePath `
  --no-warmup `
  @ExtraArgs
