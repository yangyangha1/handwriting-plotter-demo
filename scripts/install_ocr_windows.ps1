param(
    [switch]$Upgrade,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if (-not (Get-Command $Python -ErrorAction SilentlyContinue) -and -not (Test-Path -LiteralPath $Python)) {
    throw "找不到 Python：$Python。请使用 -Python 指定 Python 解释器。"
}

$upgradeArgs = @()
if ($Upgrade) {
    $upgradeArgs = @("--upgrade")
}

& $Python -m pip install @upgradeArgs "paddlepaddle==3.0.0" -i "https://www.paddlepaddle.org.cn/packages/stable/cpu/"
if ($LASTEXITCODE -ne 0) {
    throw "PaddlePaddle 安装失败，退出码：$LASTEXITCODE"
}

Push-Location $projectRoot
try {
    & $Python -m pip install @upgradeArgs -e ".[ocr]"
    if ($LASTEXITCODE -ne 0) {
        throw "PaddleOCR 安装失败，退出码：$LASTEXITCODE"
    }
} finally {
    Pop-Location
}

& $Python -c "import paddle, paddleocr; print('PaddlePaddle', paddle.__version__); print('PaddleOCR', paddleocr.__version__)"
if ($LASTEXITCODE -ne 0) {
    throw "OCR 导入检查失败，退出码：$LASTEXITCODE"
}

Write-Output "OCR 环境已就绪。"
