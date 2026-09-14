param(
    [switch]$Clean,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$dist = Join-Path $projectRoot "dist"
$build = Join-Path $projectRoot "build\pyinstaller"
$bundledData = "$(Join-Path $projectRoot 'data');data"
$bundledWeb = "$(Join-Path $projectRoot 'src\hwplotter\web');hwplotter\web"

if (-not (Get-Command $Python -ErrorAction SilentlyContinue) -and -not (Test-Path -LiteralPath $Python)) {
    throw "找不到 Python：$Python。请使用 -Python 指定 Python 解释器。"
}

& $Python -m PyInstaller --version *> $null
if ($LASTEXITCODE -ne 0) {
    & $Python -m pip install "pyinstaller>=6.0"
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller 安装失败，退出码：$LASTEXITCODE"
    }
}

if ($Clean) {
    if (Test-Path -LiteralPath $dist) { Remove-Item -LiteralPath $dist -Recurse -Force }
    if (Test-Path -LiteralPath $build) { Remove-Item -LiteralPath $build -Recurse -Force }
}

& $Python -m PyInstaller `
    --noconfirm `
    --clean `
    --console `
    --onefile `
    --name "hwplotter-web" `
    --add-data $bundledData `
    --add-data $bundledWeb `
    --paths (Join-Path $projectRoot "src") `
    --distpath $dist `
    --workpath $build `
    --specpath (Join-Path $projectRoot "build") `
    (Join-Path $projectRoot "start_web.py")

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller 构建失败，退出码：$LASTEXITCODE"
}

Write-Output "可执行文件：$(Join-Path $dist 'hwplotter-web.exe')"
