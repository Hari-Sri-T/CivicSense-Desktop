"""Optional browser check: install playwright and run `playwright install chromium`."""
import re
from urllib.parse import urlsplit

import pytest

playwright = pytest.importorskip("playwright.sync_api")
from app import app
from model_runtime import runtime


@pytest.fixture
def browser_state(monkeypatch):
    state = dict(runtime.status(), phase="ready")
    monkeypatch.setattr(runtime, "status", lambda: state.copy())
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1360, "height": 900})

        def serve(route):
            response = app.test_client().open(urlsplit(route.request.url).path, method=route.request.method)
            route.fulfill(status=response.status_code, body=response.data, headers=dict(response.headers))

        page.route("http://127.0.0.1/**", serve)
        yield page, state
        browser.close()


def test_neutral_surfaces_blue_accents_and_keyboard_text_size(browser_state):
    page, _ = browser_state
    page.goto("http://127.0.0.1/")
    page.locator("#question").focus()
    page.locator("#question").fill("Keep this question while resizing")
    font_size = lambda: float(page.locator("#question").evaluate("e => parseFloat(getComputedStyle(e).fontSize)"))
    initial = font_size()
    page.keyboard.press("Control+Equal")
    assert font_size() == pytest.approx(initial * 1.1)
    page.keyboard.press("Control+Shift+Equal")
    assert font_size() == pytest.approx(initial * 1.2)
    page.keyboard.press("Control+Minus")
    assert font_size() == pytest.approx(initial * 1.1)
    page.keyboard.press("Control+0")
    assert font_size() == initial
    page.keyboard.press("Control+NumpadAdd")
    assert font_size() == pytest.approx(initial * 1.1)
    page.keyboard.press("Meta+Equal")
    assert font_size() == pytest.approx(initial * 1.2)
    for _ in range(25):
        page.keyboard.press("Control+Equal")
    assert font_size() == pytest.approx(initial * 2)
    for _ in range(25):
        page.keyboard.press("Control+Minus")
    assert font_size() == pytest.approx(initial * .7)
    page.keyboard.press("Control+0")
    assert page.locator("#question").input_value() == "Keep this question while resizing"
    assert page.locator("#text-size-label").inner_text() == "TEXT 100%"
    for selector in ("body", "main", ".modelbar", ".activity-bar", ".composer", ".sidebar"):
        color = page.locator(selector).first.evaluate("e => getComputedStyle(e).backgroundColor")
        channels = re.findall(r"\d+", color)[:3]
        assert channels[0] == channels[1] == channels[2], color
    for selector, property_name in ((".model-selector", "backgroundColor"), (".activity.selected", "color")):
        color = page.locator(selector).first.evaluate("(e, prop) => getComputedStyle(e)[prop]", property_name)
        red, green, blue = map(int, re.findall(r"\d+", color)[:3])
        assert blue > green > red, color


def test_startup_loading_ready_and_failure_recovery(browser_state, monkeypatch):
    page, state = browser_state
    state.update(phase="loading", message="Loading Qwen into memory", error=None)
    page.goto("http://127.0.0.1/")
    page.locator("#startup-message").filter(has_text="Loading Qwen into memory").wait_for()
    assert page.locator("#startup-screen").is_visible()
    assert page.locator("#app-shell").evaluate("e => e.inert")
    assert page.locator(".startup-logo").evaluate("e => e.complete && e.naturalWidth > 0")
    state.update(phase="downloading", message="Downloading Qwen weights · 42%")
    page.evaluate("pollStatus()")
    assert "42%" in page.locator("#startup-message").inner_text()
    state.update(phase="error", message="Model setup needs attention", error="No connection")
    page.evaluate("pollStatus()")
    assert page.locator("#startup-error").inner_text() == "No connection"
    assert page.locator("#startup-retry").is_visible()
    monkeypatch.setattr(runtime, "start", lambda: state.update(phase="loading", message="Retrying model load", error=None))
    page.locator("#startup-retry").click()
    page.locator("#startup-message").filter(has_text="Retrying model load").wait_for()
    state.update(phase="ready")
    page.evaluate("pollStatus()")
    assert not page.locator("#startup-screen").is_visible()
    assert not page.locator("#app-shell").evaluate("e => e.inert")
    state.update(phase="error", error="A later error")
    page.evaluate("pollStatus()")
    assert not page.locator("#startup-screen").is_visible()
    page.reload()
    page.locator("#startup-settings").click()
    assert not page.locator("#startup-screen").is_visible()
    assert page.locator("#models-view").is_visible()
