param(
    [switch]$Clean,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$dist = Join-Path $projectRoot "dist"
$build = Join-Path $projectRoot "build\portable-launcher"

if (-not (Get-Command $Python -ErrorAction SilentlyContinue) -and -not (Test-Path -LiteralPath $Python)) {
    throw "找不到 Python：$Python。请使用 -Python 指定 Python 解释器。"
}

& $Python -m PyInstaller --version *> $null
if ($LASTEXITCODE -ne 0) {
    & $Python -m pip install "pyinstaller>=6.0"
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller 安装失败，退出码：$LASTEXITCODE" }
}

if ($Clean -and (Test-Path -LiteralPath $build)) {
    Remove-Item -LiteralPath $build -Recurse -Force
}

& $Python -m PyInstaller `
    --noconfirm `
    --clean `
    --console `
    --onefile `
    --name "handwriting-studio-portable" `
    --distpath $dist `
    --workpath $build `
    --specpath (Join-Path $projectRoot "build") `
    (Join-Path $projectRoot "portable_launcher.py")

if ($LASTEXITCODE -ne 0) { throw "便携启动器构建失败，退出码：$LASTEXITCODE" }
Write-Output "便携启动器：$(Join-Path $dist 'handwriting-studio-portable.exe')"
