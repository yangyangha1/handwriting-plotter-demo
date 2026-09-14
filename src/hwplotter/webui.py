"""Local Web UI: stdlib bootstrap works even before scientific packages are installed."""

import argparse
import base64
import importlib.util
import json
import mimetypes
import os
import secrets
import sys
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.parse import unquote, urlparse


def _runtime_mode():
    if getattr(sys, "frozen", False):
        return "frozen-exe"
    executable = Path(sys.executable).resolve()
    if (executable.parent.parent / "pyvenv.cfg").is_file():
        if executable.parent.parent.name == ".venv":
            return "project-venv"
        return "venv"
    return "system-python"


def _install_prefix():
    if _runtime_mode() == "project-venv":
        return ".venv\\Scripts\\python.exe" if os.name == "nt" else ".venv/bin/python"
    return "python"


def _copy_prefix(graphics):
    if _runtime_mode() == "project-venv":
        project = Path(graphics).resolve().parent.parent
        if os.name == "nt":
            return f'Set-Location -LiteralPath "{project}"; & ".\\.venv\\Scripts\\python.exe"'
        return f'cd "{project}"; ./.venv/bin/python'
    return "python"


def _package_installed(module, package):
    if module in {"onnxruntime_gpu", "paddle_gpu"}:
        try:
            version(package)
        except PackageNotFoundError:
            return False
        return True
    return importlib.util.find_spec(module) is not None


def environment(graphics):
    install_prefix = _install_prefix()
    copy_prefix = _copy_prefix(graphics)
    modules = (
        ("numpy", "numpy", "图像数组与数值计算"),
        ("cv2", "opencv-python", "图像读取、预处理和轮廓提取"),
        ("scipy", "scipy", "科学计算、插值和几何拟合"),
        ("skimage", "scikit-image", "图像分割、骨架和形态学处理"),
        ("PIL", "Pillow", "图片打开、保存和格式转换"),
        ("fontTools", "fonttools", "TTF 字体文件导出"),
        ("rapidocr", "rapidocr", "默认 RapidOCR 文字识别引擎"),
        ("onnxruntime", "onnxruntime", "RapidOCR CPU 推理后端"),
        ("torch", "torch", "个人模型训练和 Torch 推理，可选"),
        ("paddleocr", "paddleocr", "PaddleOCR 识别引擎，可选"),
        ("paddle", "paddlepaddle", "Paddle CPU/GPU 推理运行时，可选"),
        ("onnxruntime_gpu", "onnxruntime-gpu", "CUDA 引擎的 ONNX Runtime GPU 后端，可选"),
        ("paddle_gpu", "paddlepaddle-gpu", "GPU 引擎的 PaddlePaddle GPU 后端，可选"),
    )
    rows = [
        {
            "module": m,
            "package": p,
            "purpose": purpose,
            "installed": _package_installed(m, p),
            "command": f"{install_prefix} -m pip install {p}",
            "copy_command": f"{copy_prefix} -m pip install {p}",
        }
        for m, p, purpose in modules
    ]
    core_modules = {
        "numpy",
        "cv2",
        "scipy",
        "skimage",
        "PIL",
        "fontTools",
        "rapidocr",
        "onnxruntime",
    }
    runtime_mode = _runtime_mode()
    executable = Path(sys.executable).resolve()
    resolved_data = Path(graphics).resolve()
    non_ascii_path = any(ord(char) > 127 for char in str(resolved_data))
    return {
        "python": sys.version.split()[0],
        "data_dir": str(resolved_data.parent),
        "supported_python": sys.version_info >= (3, 10),
        "core_ready": all(x["installed"] for x in rows if x["module"] in core_modules),
        "dependencies": rows,
        "graphics_available": resolved_data.is_file(),
        "data_command": f"{install_prefix} scripts/download_mmh.py",
        "data_copy_command": f"{copy_prefix} scripts/download_mmh.py",
        "core_command": f'{install_prefix} -m pip install -e "."',
        "dictionary_available": resolved_data.with_name("dictionary.txt").is_file(),
        "python_executable": str(executable),
        "runtime_mode": runtime_mode,
        "environment_root": str(
            executable.parent if runtime_mode == "frozen-exe" else executable.parent.parent
        ),
        "note": "检测安装状态；模型权重首次OCR时下载，下载/运行错误显示在任务结果。",
        "path_ascii_only": not non_ascii_path,
        "path_warning": (
            "检测到项目或数据路径包含中文/非 ASCII 字符。程序已启用兼容图片读写；"
            "如仍出现图片读取错误，请将整个项目移动到只含英文字母、数字和短横线的目录。"
            if non_ascii_path
            else ""
        ),
    }


def describe_error(exc):
    text = str(exc).strip()
    if isinstance(exc, FileNotFoundError):
        return f"文件读取失败：{text or '文件不存在'}"
    if isinstance(exc, ModuleNotFoundError):
        return f"依赖模块缺失：{text or '请检查本地环境'}"
    if isinstance(exc, ImportError):
        return f"依赖加载失败：{text or '请检查本地环境'}"
    return f"{type(exc).__name__}：{text}" if text else type(exc).__name__


def log_error(exc):
    message = describe_error(exc)
    print(f"[Handwriting Studio] {message}", file=sys.stderr, flush=True)
    return message


def make_server(root, graphics, port=8765):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    executor = ThreadPoolExecutor(max_workers=1)
    jobs = {}
    studio = None
    token = secrets.token_urlsafe(24)

    def service():
        nonlocal studio
        if studio is None:
            from hwplotter.web_service import Studio

            studio = Studio(root, graphics)
        return studio

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, data, kind="application/json"):
            body = (
                json.dumps(data, ensure_ascii=False).encode()
                if kind == "application/json"
                else data
            )
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            try:
                path = urlparse(self.path).path
                if path == "/":
                    b = (
                        (Path(__file__).parent / "web/index.html")
                        .read_text("utf-8")
                        .replace("__TOKEN__", token)
                    )
                    return self.send(200, b.encode(), "text/html; charset=utf-8")
                if path == "/api/environment":
                    return self.send(200, environment(graphics))
                if path == "/api/state":
                    return self.send(200, service().state)
                if path.startswith("/api/jobs/"):
                    f = jobs.get(path.rsplit("/", 1)[-1])
                    if f is None:
                        return self.send(404, {"error": "任务不存在"})
                    if not f.done():
                        return self.send(200, {"status": "running", "progress": service().progress})
                    try:
                        return self.send(200, {"status": "done", "result": f.result()})
                    except Exception as e:  # noqa: BLE001 - HTTP/job error response
                        return self.send(
                            200,
                            {
                                "status": "error",
                                "error": log_error(e),
                                "error_type": type(e).__name__,
                            },
                        )
                if path.startswith("/files/"):
                    p = (root / unquote(path[7:])).resolve()
                    if not p.is_relative_to(root) or not p.is_file():
                        return self.send(404, {"error": "文件不存在"})
                    return self.send(
                        200,
                        p.read_bytes(),
                        mimetypes.guess_type(p.name)[0] or "application/octet-stream",
                    )
                return self.send(404, {"error": "未找到"})
            except Exception as e:  # noqa: BLE001 - HTTP/job error response
                self.send(400, {"error": log_error(e), "error_type": type(e).__name__})

        def do_POST(self):
            if self.headers.get("X-Studio-Token") != token:
                return self.send(403, {"error": "请求验证失败，请刷新页面"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if not 0 < n <= 30_000_000:
                    raise ValueError("上传请求大小须小于30MB")
                data = json.loads(self.rfile.read(n))
                action = urlparse(self.path).path
                if action == "/api/engine-check":
                    return self.send(200, service().check_engine(str(data.get("engine", "rapid"))))
                if any(not f.done() for f in jobs.values()):
                    return self.send(409, {"error": "已有任务正在运行"})

                def run():
                    s = service()
                    s.progress = {}
                    if action == "/api/upload":
                        return s.ingest(
                            base64.b64decode(data["image"], validate=True),
                            int(data.get("threshold", 0)),
                        )
                    if action == "/api/ocr":
                        return s.recognize(data["id"], data.get("engine", "rapid"))
                    if action == "/api/learn":
                        return s.learn(data["id"], data["indices"])
                    if action == "/api/train":
                        return s.train_legacy(data["id"], data["indices"])
                    if action == "/api/generate":
                        allowed = {
                            key: data[key]
                            for key in (
                                "text",
                                "draft",
                                "all_chars",
                                "gap",
                                "line_height",
                                "page_mm",
                                "pen_up",
                                "pen_down",
                                "feed",
                                "use_model",
                            )
                            if key in data
                        }
                        return s.generate(**allowed)
                    if action == "/api/font-package":
                        return s.generate_font_package(data.get("kind", "observed"))
                    raise ValueError("未知操作")

                jid = secrets.token_hex(8)
                jobs[jid] = executor.submit(run)
                return self.send(202, {"job": jid})
            except Exception as e:  # noqa: BLE001 - HTTP/job error response
                self.send(400, {"error": log_error(e), "error_type": type(e).__name__})

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    server.block_on_close = False
    server.executor = executor
    return server


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--work-dir", default="out/web")
    p.add_argument("--graphics", default="data/graphics.txt")
    p.add_argument("--no-browser", action="store_true")
    a = p.parse_args()
    server = make_server(a.work_dir, a.graphics, a.port)
    url = f"http://127.0.0.1:{server.server_port}"
    print("Handwriting Studio " + url, flush=True)
    if not a.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.executor.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    main()
