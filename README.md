# Handwriting Studio v1.2

**可运行的 Web 工作台，尚未达到“少量手稿完美生成全套个人草书”的验收标准。**

已实现：整页墨迹轮廓 SVG、保留原稿间距、图遍历中心路径、OCR 与逐字核对、真实字样字库、TTF、整段 SVG/G-code、逐字路径 JSONL、全部 MMH 字符批量导出、原有层级模型训练入口、首次环境检测。

最终字库输出明确分为两部分：

- **已有字样完整字库**：汇总本次会话中所有已确认的真实字样，不受输入测试句子限制；即使模型数据量不足，也可以导出这部分真实字形。
- **所有完整字库**：在模型训练状态达到 `READY`、样本覆盖和独立验证均满足要求时，才允许导出推测字形的完整字库。模型处理完成但数据量不足时，界面会明确显示“模型已处理完成，完整字库不可用”，不会把处理完成误认为可用。

本版本以 v0.9 为功能基线，保留其字体导出、Web 服务、验证证据和安全边界；对 v0.8 与 v0.9 的同名内容以 v0.9 为准，并保留 v0.8 的环境检查交互风格。

## 推荐启动：项目自带虚拟环境

推荐双击 `dist/handwriting-studio-portable.exe`，或在 PowerShell 7 中执行：

```powershell
.\dist\handwriting-studio-portable.exe
```

程序会按顺序执行：

1. 首先检测系统中是否有 Python 3.10 或更高版本；
2. 在当前项目目录创建 `.venv`，不使用用户主 Python 安装包；
3. 使用项目内的 `requirements-portable.txt` 安装核心 CPU/RapidOCR 运行依赖；Torch、PaddleOCR、GPU/CUDA 后端保持可选，不自动安装；
4. 将 pip、OCR/AI 模型缓存和运行缓存放在项目的 `.cache` 内；
5. 从当前项目的 `data/graphics.txt` 和 `data/dictionary.txt` 读取工程数据；
6. 启动本地 Web 页面。

两个 MMH txt 文件已经随源文件和压缩包提供；如果用户误删，启动器会尝试自动从官方地址重新创建。依赖安装失败或缺少非关键依赖时不会退出，仍会打开环境检查页面，用户可以在页面中查看缺失项。首次启动会下载依赖，可能需要较长时间和网络连接。以后再次启动会检查清单哈希，依赖未变化时不会重复安装。删除整个项目文件夹即可一并删除 `.venv`、缓存和运行数据，不影响系统 Python 或其他项目。

未实现/未通过：任意草书自动正确识别；真实书写时序恢复；未见字个人风格保真；无限 Unicode 字符覆盖；写字机硬件实测。实验字形会在界面和 coverage.json 明确标注，不用标准楷书冒充本人书写。

## 启动（Python 3.10+）

如果不使用便携启动器，也可以先进入解压目录，再执行：

```bash
python start_web.py
```

此入口只用 Python 标准库，**未安装 numpy/OpenCV 等也能打开环境检测页**。浏览器打开 `http://127.0.0.1:8765`，页面列出缺失包、当前 Python 解释器对应的安装命令及笔画数据状态。Python 本身需要先安装。

手动创建环境：

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
python -m pip install -e ".[rapidocr,ml,dev]"
python start_web.py
```

Windows 如需构建单文件程序：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_web_exe.ps1 -Clean
```

脚本会把 `data/` 和 Web 页面资源一并打包，并输出 `dist/hwplotter-web.exe`。

本交付包不预置旧的单文件 Web 程序，避免用户误以为系统 Python 安装可以改变打包环境。若需要传统单文件程序，可在依赖齐全的构建环境中执行上面的构建脚本；日常使用请启动 `handwriting-studio-portable.exe`。

便携启动器默认浏览器打开 `http://127.0.0.1:8765`；如不希望自动打开浏览器，可执行 `.\dist\handwriting-studio-portable.exe --no-browser`。关闭命令窗口即可停止本地服务。

手动启动时需要安装运行时模型依赖：`python -m pip install -e ".[ml,ocr]"`，Windows 还需按运行平台安装 PaddlePaddle；便携启动器默认不自动安装 Torch、PaddleOCR、GPU/CUDA 等可选模块。开发工具和旧版 GUI 不属于便携运行依赖。
PaddleOCR 可选：`python -m pip install -e ".[ocr]"`，另按运行平台安装 PaddlePaddle。
默认 RapidOCR 使用 CPU / ONNX；首次识别需要下载模型，失败会显示具体错误。

Windows CPU 如需安装 PaddleOCR，可使用：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_ocr_windows.ps1
```

## 流程

1. 上传自然手写照片，生成原稿轮廓；旧纸可调整阈值（本次古帖测试使用85）。界面可切换原图、轮廓、中心路径和二值墨迹。
2. 选择 OCR 引擎识别。逐字核对图像和标签，**仅勾选两者均正确的样本**。识别漏字可能导致整行后续切块错位，不能只看置信度。
3. 保存真实字样。可选“训练原有层级模型”，查看有效样本、补字建议和 NEED_MORE_SAMPLES 状态。
4. 输入新文字。未提供样本的字默认拒绝；勾选“未写字实验估计”后可生成基础笔画经几何统计调整的结果。选择“使用已训练层级模型”可调用既有 v0.8 模型链；它同样不代表通过个人草书验收。
5. 下载 TTF、SVG、G-code 或完整字库路径包。完整包包含 `glyphs.jsonl`：每字的轮廓、中心路径及来源。TTF 是轮廓字体，本身不是单线笔顺文件。
6. “全部字”明确指本包 MMH 数据的9574个字符；不是所有汉字/标点/拉丁字母。数据不支持的字符报错，不静默替换。

原稿模式保留图像间距；重排模式使用统一字面和可调字间距/行距。没有实现从少量文章学习全套标点避让、跨字连笔或上下文替代字形。一个字符目前使用最近确认的样本，原有多变体轨迹库仍保留在层级流程中。

## 查看已测试的例子

```bash
python start_web.py --work-dir examples/verified_session
```

`evidence/web_ui.png`、`web_generated.png`、`web_mobile.png` 是实际 Chromium 截图。`examples/full_font_bundle.zip` 是实测9574字的**实验字库**（含字体和路径），不能作为该书写者的完整个人字体成品。

上述会话目录和实验字库是本地验证生成物，不纳入源代码仓库；运行命令或从发布压缩包启动后即可重新生成。源仓库保留程序、测试、数据授权文件和必要的界面验收证据。

## 测试

```bash
python -m pytest -q
python -m ruff check src tests
python -m mypy src --ignore-missing-imports --no-incremental
python -m compileall -q src
```

浏览器测试额外安装 `python -m pip install playwright`、`python -m playwright install chromium` 后执行：

```bash
python scripts/browser_e2e.py
```

已有浏览器时设置 `CHROMIUM_EXECUTABLE`。脚本自建临时本地服务器，从文件上传一路点击到 TTF 实际渲染，并检查手机宽度无横向溢出。固定样稿正确索引由人工核对，只对这个测试图片有效，不是生产硬编码。

## 工程边界

服务仅监听本机127.0.0.1，上传/操作使用会话令牌，文件访问限制在工作目录。耗时任务串行执行；关闭进程会中断未完成任务。字体逐字落盘以控制批量生成内存。

G-code 坐标沿图像方向（Y向下），默认笔控 `M3 S0`/`M3 S1000`。重排输出可修改笔控、页面宽度和速度；原稿快捷下载使用默认180mm宽度。页面高度按比例变化，未连接设备、未验证行程/压力；应在实际控制软件中确认坐标与笔控后使用。

源数据和许可证见 `THIRD_PARTY.md` 与 `data/`。本次版本整合记录见 `docs/MERGE_NOTES_v1.2.md`，验收记录见 `docs/AUDIT_v1.2.md`。旧版说明保留在 `docs/README_v0.8_history.md`，其中的“高置信”“完成”等历史措辞不能作为当前验收结论。
