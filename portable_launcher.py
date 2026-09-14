"""Create and run the project-local Python environment.

This launcher is intentionally standard-library-only so it can run before the
project dependencies exist.  A frozen copy can be distributed as the primary
entry point for Windows users; it invokes the project-local venv after the
first-run checks and dependency installation complete.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen

MIN_PYTHON = (3, 10)
REQUIREMENTS_NAME = "requirements-portable.txt"
MARKER_NAME = ".hwplotter-dependencies.json"
REQUIRED_MODULES = (
    "numpy",
    "cv2",
    "scipy",
    "skimage",
    "PIL",
    "fontTools",
    "rapidocr",
    "onnxruntime",
)
MMH_BASE_URL = "https://raw.githubusercontent.com/skishore/makemeahanzi/master"
MMH_FILES = ("graphics.txt", "dictionary.txt")


def project_root() -> Path:
    if getattr(sys, "frozen", False):
        executable_dir = Path(sys.executable).resolve().parent
        if (executable_dir / REQUIREMENTS_NAME).is_file():
            return executable_dir
        if (executable_dir.parent / REQUIREMENTS_NAME).is_file():
            return executable_dir.parent
        return executable_dir
    return Path(__file__).resolve().parent


def log(message: str) -> None:
    print(f"[Handwriting Studio] {message}", flush=True)


def _candidate_commands() -> list[list[str]]:
    candidates: list[list[str]] = []
    if not getattr(sys, "frozen", False):
        candidates.append([sys.executable])
    if os.name == "nt":
        py = shutil.which("py")
        if py:
            candidates.extend([[py, "-3.12"], [py, "-3"]])
    for name in ("python", "python3"):
        executable = shutil.which(name)
        if executable:
            candidates.append([executable])
    unique: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for command in candidates:
        key = tuple(command)
        if key not in seen:
            seen.add(key)
            unique.append(command)
    return unique


def detect_python(root: Path) -> Path:
    probe = (
        "import json,sys; "
        "print(json.dumps({'major':sys.version_info.major,'minor':sys.version_info.minor,"
        "'executable':sys.executable}))"
    )
    failures: list[str] = []
    for command in _candidate_commands():
        try:
            result = subprocess.run(
                [*command, "-c", probe],
                cwd=root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            failures.append(f"{' '.join(command)}: {exc}")
            continue
        if result.returncode != 0:
            failures.append(f"{' '.join(command)}: {result.stderr.strip()}")
            continue
        try:
            info = json.loads(result.stdout.strip())
        except json.JSONDecodeError:
            failures.append(f"{' '.join(command)}: 无法读取版本信息")
            continue
        version = (int(info["major"]), int(info["minor"]))
        if version < MIN_PYTHON:
            failures.append(f"{' '.join(command)}: Python {version[0]}.{version[1]} 过低")
            continue
        return Path(str(info["executable"])).resolve()
    detail = "\n".join(failures[-4:])
    raise RuntimeError(
        "未找到可用的 Python 3.10 或更高版本。请先安装 Python，并勾选“Add Python to PATH”。"
        + (f"\n检测记录：\n{detail}" if detail else "")
    )


def venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def run(command: list[str], root: Path, env: dict[str, str] | None = None) -> None:
    log("执行：" + " ".join(f'"{x}"' if " " in x else x for x in command))
    subprocess.run(command, cwd=root, env=env, check=True)


def local_environment(root: Path) -> dict[str, str]:
    cache = root / ".cache"
    env = os.environ.copy()
    env.update(
        {
            "PIP_CACHE_DIR": str(cache / "pip"),
            "TORCH_HOME": str(cache / "torch"),
            "HF_HOME": str(cache / "huggingface"),
            "PADDLE_HOME": str(cache / "paddle"),
            "XDG_CACHE_HOME": str(cache),
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        }
    )
    return env


def ensure_venv(root: Path, base_python: Path) -> Path:
    venv = root / ".venv"
    python = venv_python(venv)
    if python.is_file():
        return python
    if venv.exists():
        raise RuntimeError(f"项目内虚拟环境不完整：{venv}\n请删除该 .venv 文件夹后重新启动。")
    log(f"正在项目文件夹内创建虚拟环境：{venv}")
    run([str(base_python), "-m", "venv", str(venv)], root)
    if not python.is_file():
        raise RuntimeError(f"虚拟环境创建失败，未找到：{python}")
    return python


def ensure_project_data(root: Path) -> None:
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    missing = [name for name in MMH_FILES if not (data_dir / name).is_file()]
    if not missing:
        log("项目 data 文件已找到：graphics.txt、dictionary.txt。")
        return
    log("项目 data 文件不完整，尝试自动创建缺少的 MMH txt 文件…")
    for name in missing:
        target = data_dir / name
        temporary = target.with_suffix(target.suffix + ".download")
        try:
            with urlopen(f"{MMH_BASE_URL}/{name}", timeout=60) as response, temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            temporary.replace(target)
            log(f"已创建：{target}")
        except (OSError, TimeoutError) as exc:
            if temporary.exists():
                temporary.unlink()
            log(f"警告：无法自动创建 {name}（{exc}）。继续打开环境检查页面。")


def requirements_changed(root: Path, venv: Path) -> bool:
    requirements = root / REQUIREMENTS_NAME
    marker = venv / MARKER_NAME
    if not requirements.is_file():
        log(f"警告：缺少依赖清单 {requirements}，跳过自动安装并继续启动。")
        return False
    digest = hashlib.sha256(requirements.read_bytes()).hexdigest()
    try:
        saved = json.loads(marker.read_text("utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        saved = {}
    return saved.get("requirements_sha256") != digest


def write_dependency_marker(root: Path, venv: Path, python: Path) -> None:
    requirements = root / REQUIREMENTS_NAME
    digest = hashlib.sha256(requirements.read_bytes()).hexdigest()
    marker = venv / MARKER_NAME
    marker.write_text(
        json.dumps({"requirements_sha256": digest, "python": str(python)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def required_modules_present(root: Path, python: Path) -> bool:
    probe = (
        "import importlib.util,json; "
        f"print(json.dumps({{name: importlib.util.find_spec(name) is not None for name in {REQUIRED_MODULES!r}}}))"
    )
    result = subprocess.run(
        [str(python), "-c", probe],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        return False
    try:
        status = json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        return False
    return all(bool(status.get(name)) for name in REQUIRED_MODULES)


def install_dependencies(root: Path, python: Path) -> None:
    venv = python.parent.parent
    requirements = root / REQUIREMENTS_NAME
    if not requirements.is_file():
        log("非关键依赖清单缺失，继续打开环境检查页面。")
        return
    changed = requirements_changed(root, venv)
    ready = required_modules_present(root, python)
    marker = venv / MARKER_NAME
    if ready and not changed:
        log("项目依赖已安装，跳过重复安装。")
        return
    if ready and not marker.is_file():
        write_dependency_marker(root, venv, python)
        log("已检测到项目依赖齐全，跳过安装并记录状态。")
        return
    env = local_environment(root)
    log("检测到项目虚拟环境缺少依赖或依赖清单已变化，开始补齐；不会修改主 Python。")
    try:
        run([str(python), "-m", "pip", "install", "-r", str(requirements)], root, env)
    except (OSError, subprocess.CalledProcessError) as exc:
        log(f"警告：部分依赖安装失败（{exc}）。继续打开环境检查页面，可稍后逐项处理。")
        return
    if required_modules_present(root, python):
        write_dependency_marker(root, venv, python)
        log("项目依赖补齐完成。")
    else:
        log("警告：安装命令结束，但仍有依赖缺失；继续打开环境检查页面。")


def launch(root: Path, python: Path, arguments: list[str]) -> int:
    command = [str(python), str(root / "start_web.py"), *arguments]
    if "--work-dir" not in arguments:
        command.extend(["--work-dir", str(root / "out" / "portable_session")])
    if "--graphics" not in arguments:
        command.extend(["--graphics", str(root / "data" / "graphics.txt")])
    log(f"使用项目虚拟环境启动：{python}")
    process = subprocess.Popen(command, cwd=root, env=local_environment(root))
    try:
        return process.wait()
    except KeyboardInterrupt:
        log("正在关闭本地 Web 服务并释放项目资源…")
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        return 130


def main() -> int:
    root = project_root()
    try:
        log("第 1 步：检测 Python 环境…")
        base_python = detect_python(root)
        log(f"检测到 Python：{base_python}")
        log(f"项目目录：{root}")
        if "--check-python" in sys.argv[1:]:
            return 0
        ensure_project_data(root)
        python = ensure_venv(root, base_python)
        install_dependencies(root, python)
        arguments = [arg for arg in sys.argv[1:] if arg != "--check-python"]
        return launch(root, python, arguments)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"\n启动失败：{exc}", file=sys.stderr, flush=True)
        if getattr(sys, "frozen", False) and os.name == "nt":
            input("按 Enter 键关闭窗口…")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
