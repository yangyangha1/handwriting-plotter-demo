"""Real UI acceptance. Provide CHROMIUM_EXECUTABLE if a browser is already installed."""

from pathlib import Path
import json, os, threading, tempfile
from hwplotter.webui import make_server
from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parents[1]
server = make_server(
    Path(tempfile.mkdtemp(prefix="browser_", dir=root / "out")), root / "data/graphics.txt", 0
)
threading.Thread(target=server.serve_forever, daemon=True).start()
with sync_playwright() as pw:
    browser = pw.chromium.launch(
        executable_path=os.environ.get("CHROMIUM_EXECUTABLE"),
        headless=True,
        args=[
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--use-gl=angle",
            "--use-angle=swiftshader",
        ],
    )
    page = browser.new_page(viewport={"width": 1440, "height": 1050}, device_scale_factor=1)
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"http://127.0.0.1:{server.server_port}")
    page.wait_for_function(
        "document.querySelector('#commands').textContent.includes('pip install')"
    )
    page.locator("#file").set_input_files(root / "evidence/source_0.jpg")
    page.locator("#threshold").fill("85")
    print("upload", flush=True)
    page.locator("#upload").click()
    page.wait_for_function("document.querySelector('#source').naturalWidth>0", timeout=120000)
    print("ocr", flush=True)
    page.locator("#ocr").click()
    page.wait_for_selector("#review input", timeout=180000)
    page.wait_for_function("document.querySelector('#status').textContent.includes('OCR完成')")
    # Human-reviewed indices for the fixed public fixture; no OCR ground-truth injection.
    for i in [13, 14, 15, 16, 17, 18, 19, 20, 21, 22]:
        page.locator(f'#review input[value="{i}"]').check()
    print("learn", flush=True)
    page.locator("#learn").click()
    page.wait_for_function(
        "document.querySelector('#status').textContent.includes('累计')", timeout=60000
    )
    page.locator("#text").fill("快雪時晴佳想\n山陰張侯 天地玄黃")
    page.locator("#generate").click()
    page.wait_for_function(
        "document.querySelector('#status').textContent.includes('未提供字样')", timeout=30000
    )
    print("generate", flush=True)
    page.locator("#draft").check()
    page.locator("#generate").click()
    page.wait_for_function(
        "document.querySelector('#fontText').textContent.length>0", timeout=120000
    )
    page.wait_for_function("document.querySelector('#genSvg').naturalWidth>0")
    page.screenshot(path=root / "evidence/web_ui.png", full_page=True)
    page.locator("#generated").screenshot(path=root / "evidence/web_generated.png")
    report = json.loads(page.locator("#report").text_content())
    assert report["observed_count"] == 10 and report["experimental_count"] == 4
    assert not errors, errors
    page.set_viewport_size({"width": 390, "height": 844})
    page.screenshot(path=root / "evidence/web_mobile.png", full_page=True)
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    (root / "evidence/browser_result.json").write_text(
        json.dumps(
            {
                "console_errors": errors,
                "report": report,
                "desktop": True,
                "mobile_no_overflow": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("Browser E2E passed", flush=True)
    browser.close()

server.shutdown()
server.server_close()
server.executor.shutdown()
