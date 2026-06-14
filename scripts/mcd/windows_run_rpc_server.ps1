param(
  [string]$Root = "C:\Users\-_-kc\Desktop\llama.cpp",
  [string]$BuildDir = "C:\Users\-_-kc\Desktop\llama.cpp\build-mcd-cuda",
  [string]$HostAddress = "169.254.21.157",
  [int]$Port = 50052,
  [string]$Device = ""
)

$ErrorActionPreference = "Stop"
$Bin = Join-Path $BuildDir "bin\Release\rpc-server.exe"
$Args = @("--host", $HostAddress, "--port", $Port)
if ($Device -ne "") {
  $Args += @("--device", $Device)
}

& $Bin @Args
