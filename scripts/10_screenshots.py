# Capture dashboard screenshots for the README, reproducibly.
#
#   python scripts/10_screenshots.py
#
# Requires both servers running:
#   uvicorn backend.main:app --port 8000
#   npm run dev --prefix frontend
#
# One-time setup:
#   pip install playwright && playwright install chromium
#
# Output: docs/screenshots/*.png

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"
URL = os.environ.get("DASHBOARD_URL", "http://localhost:3000")
API = os.environ.get("API_URL", "http://localhost:8000")
WIDTH = int(os.environ.get("SHOT_WIDTH", "1600"))
HEIGHT = int(os.environ.get("SHOT_HEIGHT", "900"))

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    raise SystemExit("playwright missing.  pip install playwright && playwright install chromium")

import urllib.request

for name, url in (("API", API + "/health"), ("dashboard", URL)):
    try:
        urllib.request.urlopen(url, timeout=5)
    except Exception as exc:
        raise SystemExit(f"{name} not reachable at {url} ({exc}).  Start both servers first.")

OUT.mkdir(parents=True, exist_ok=True)


def wait_for_map(page, timeout=40000):
    """The map is WebGL: wait until it has actually drawn sign layers."""
    page.wait_for_selector('[data-testid="map"] canvas', timeout=timeout)
    page.wait_for_function(
        """() => {
            const el = document.querySelector('[data-testid="panel"]');
            return el && /\\d/.test(el.innerText) && !el.innerText.includes('Loading');
        }""",
        timeout=timeout,
    )
    page.wait_for_timeout(3500)   # let tiles and cluster labels settle


with sync_playwright() as pw:
    browser = pw.chromium.launch()
    page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT},
                            device_scale_factor=2)

    print("loading", URL)
    page.goto(URL, wait_until="networkidle")
    wait_for_map(page)

    # 1. statewide overview with the inventory summary open
    page.screenshot(path=str(OUT / "01-overview.png"))
    print("  01-overview.png")

    # 2. the same view without the panel, so the map reads clearly
    page.click('[data-testid="panel-close"]')
    page.wait_for_timeout(1200)
    page.screenshot(path=str(OUT / "02-map.png"))
    print("  02-map.png")

    # 3. a single sign: crop, class, confidence, accuracy vs ground truth
    page.click("text=DETAILS")
    page.wait_for_timeout(800)
    page.locator('[data-testid="detection-item"]').first.click()
    page.wait_for_timeout(4000)      # easeTo + tile load at zoom 17
    page.screenshot(path=str(OUT / "03-sign-detail.png"))
    print("  03-sign-detail.png")

    browser.close()

print("\nsaved to", OUT)
for f in sorted(OUT.glob("*.png")):
    print(f"  {f.name:24s} {f.stat().st_size/1024:6.0f} KB")
