param(
  [string]$Root = "C:\Users\-_-kc\Desktop\llama.cpp",
  [string]$BuildDir = "C:\Users\-_-kc\Desktop\llama.cpp\build-mcd-cuda"
)

$ErrorActionPreference = "Stop"
Set-Location $Root

cmake -S . -B $BuildDir `
  -DGGML_CUDA=ON `
  -DGGML_RPC=ON `
  -DCMAKE_BUILD_TYPE=Release

cmake --build $BuildDir `
  --config Release `
  --target llama-server llama-cli llama-bench rpc-server `
  -j $env:NUMBER_OF_PROCESSORS

Write-Output "build_dir=$BuildDir"
