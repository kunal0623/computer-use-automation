"""Web surface abstraction for the BankGPT computer-use agent.

Wraps Playwright behind a small, stable API: start a browser, perceive the
page as a structured observation, and act on it through a strategy chain of
element locators (role, label, placeholder, css, xpath, text) so a single
fragile selector never sinks a run. No em dashes are used anywhere in this file.
"""

from __future__ import annotations

from typing import Any, Literal

from playwright.sync_api import Locator, Page, sync_playwright
from pydantic import BaseModel, Field


class SurfaceState(BaseModel):
    """Structured observation of the current page."""

    model_config = {"arbitrary_types_allowed": True}

    url: str
    title: str
    a11y_tree: str
    dom_excerpt: str
    screenshot_png: bytes | None = Field(default=None, exclude=True)


class TargetNotFoundError(Exception):
    """Raised when no strategy in the chain resolves to a visible element."""


StrategyType = Literal["css", "xpath", "text", "role", "placeholder", "label"]

_STRATEGY_ORDER: tuple[StrategyType, ...] = ("role", "label", "placeholder", "css", "xpath", "text")


def build_strategy_chain(hints: dict | None, page: Page) -> list[dict]:
    """Convert LLM target hints into an ordered, most-robust-first strategy chain.

    Order: role (needs role+name), label, placeholder, css, xpath, text.
    Returns [] when hints is None. No rationale is attached here; the agent
    adds that when it records the step.
    """
    if not hints:
        return []
    chain: list[dict] = []

    def _get(*keys: str) -> Any | None:
        for key in keys:
            if key in hints and hints[key] not in (None, ""):
                return hints[key]
        return None

    role = _get("role")
    name = _get("name")
    if role and name:
        chain.append({"type": "role", "role": role, "name": name})

    label = _get("label")
    if label:
        chain.append({"type": "label", "value": label})

    placeholder = _get("placeholder")
    if placeholder:
        chain.append({"type": "placeholder", "value": placeholder})

    css = _get("css")
    if css:
        chain.append({"type": "css", "value": css})

    xpath = _get("xpath")
    if xpath:
        chain.append({"type": "xpath", "value": xpath})

    text = _get("text")
    if text:
        chain.append({"type": "text", "value": text})

    return chain


def _locator_for(page: Page, strategy: dict) -> Locator:
    stype = strategy.get("type")
    value = strategy.get("value")
    if stype == "css":
        return page.locator(value)
    if stype == "xpath":
        return page.locator(f"xpath={value}")
    if stype == "text":
        return page.get_by_text(value, exact=False)
    if stype == "role":
        return page.get_by_role(strategy["role"], name=strategy.get("name"))
    if stype == "placeholder":
        return page.get_by_placeholder(value)
    if stype == "label":
        return page.get_by_label(value)
    raise ValueError(f"Unknown strategy type: {stype!r}")


def resolve_strategies(page: Page, strategies: list[dict]) -> tuple[Locator, int]:
    """Try each strategy in order; first visible match wins.

    Returns (locator, index_of_winning_strategy). Raises TargetNotFoundError
    listing the tried strategies when nothing visible resolves.
    """
    if not strategies:
        raise TargetNotFoundError("No strategies provided to resolve.")
    tried: list[str] = []
    for index, strategy in enumerate(strategies):
        tried.append(f"{index}:{strategy.get('type')}={strategy.get('value') or strategy.get('name')}")
        locator = _locator_for(page, strategy)
        try:
            count = locator.count()
        except Exception:
            continue
        for n in range(count):
            try:
                candidate = locator.nth(n)
                if candidate.is_visible():
                    return candidate, index
            except Exception:
                continue
    raise TargetNotFoundError(
        "No visible element found. Tried strategies: " + "; ".join(tried)
    )


def _render_a11y(node: Any, depth: int = 0) -> list[str]:
    """Render a Playwright accessibility snapshot node to indented text lines."""    """Render a Playwright accessibility snapshot node to indented text lines."""
    lines: list[str] = []
    if isinstance(node, dict):
        role = node.get("role", "?")
        name = node.get("name", "")
        value = node.get("value")
        checked = node.get("checked")
        parts = [f"[{role}]"]
        if name:
            parts.append(f'"{name}"')
        if value not in (None, ""):
            parts.append(f"value={value!r}")
        if checked not in (None, False):
            parts.append(f"checked={checked}")
        lines.append("  " * depth + " ".join(parts))
        children = node.get("children") or []
        for child in children:
            lines.extend(_render_a11y(child, depth + 1))
    elif isinstance(node, list):
        for child in node:
            lines.extend(_render_a11y(child, depth))
    return lines


_FIND_CONTROL_JS = """el => {
  const xpathOf = (node) => {
    if (node.id) return `//*[@id="${node.id}"]`;
    const parts = [];
    let cur = node;
    while (cur && cur.nodeType === 1) {
      let i = 1;
      let sib = cur.previousSibling;
      while (sib) {
        if (sib.nodeType === 1 && sib.nodeName === cur.nodeName) i++;
        sib = sib.previousSibling;
      }
      parts.unshift(`${cur.nodeName.toLowerCase()}[${i}]`);
      cur = cur.parentNode;
    }
    return '/' + parts.join('/');
  };
  const isEditable = (n) => {
    if (!n || n.nodeType !== 1) return false;
    const tag = n.tagName.toLowerCase();
    if (tag === 'input' || tag === 'textarea' || tag === 'select') {
      return !n.disabled && !n.readOnly;
    }
    return !!n.isContentEditable;
  };
  if (isEditable(el)) return null;
  const seen = new Set();
  const candidates = [];
  const pushAll = (root) => {
    if (!root || !root.querySelectorAll) return;
    root.querySelectorAll('input, textarea, select').forEach((n) => {
      if (!seen.has(n) && isEditable(n)) { seen.add(n); candidates.push(n); }
    });
  };
  const labels = [];
  if (el.tagName === 'LABEL') labels.push(el);
  const wrapped = el.closest('label');
  if (wrapped && wrapped !== el) labels.push(wrapped);
  const elId = el.getAttribute && el.getAttribute('id');
  if (elId && window.CSS && CSS.escape) {
    document.querySelectorAll(`label[for="${CSS.escape(elId)}"]`).forEach((l) => labels.push(l));
  }
  for (const l of labels) {
    if (l.control && isEditable(l.control)) return xpathOf(l.control);
  }
  pushAll(el.parentElement);
  if (candidates.length) return xpathOf(candidates[0]);
  let sib = el.nextElementSibling;
  let hops = 0;
  while (sib && hops < 4 && !candidates.length) {
    pushAll(sib); sib = sib.nextElementSibling; hops++;
  }
  sib = el.previousElementSibling;
  hops = 0;
  while (sib && hops < 4 && !candidates.length) {
    pushAll(sib); sib = sib.previousElementSibling; hops++;
  }
  if (candidates.length) return xpathOf(candidates[0]);
  const scope = el.closest('form, fieldset, section, div, td, li');
  pushAll(scope);
  if (candidates.length) return xpathOf(candidates[0]);
  return null;
}"""


_READ_TEXT_JS = """el => {
  const text = (n) => ((n && n.innerText) || '').trim();
  const own = text(el);
  let node = el.parentElement;
  while (node && node !== document.body) {
    const t = text(node);
    if (t.length > own.length && t.length <= 800) return t;
    node = node.parentElement;
  }
  return own;
}"""


def _editable_locator(page: Page, locator: Locator) -> Locator:
    """Map a resolved locator to its editable form control when needed.

    Text hints such as {"text": "Username"} usually resolve to a <label>
    rather than the <input> itself, and Playwright's fill/select require the
    control. This keeps the reported strategy chain intact and only swaps the
    element we act on.
    """
    try:
        xpath = locator.evaluate(_FIND_CONTROL_JS)
    except Exception:
        return locator
    if xpath:
        return page.locator(f"xpath={xpath}")
    return locator


def editable_locator(page: Page, locator: Locator) -> Locator:
    """Public alias of the label-to-control mapping for replay use."""
    return _editable_locator(page, locator)


def readable_text(page: Page, locator: Locator) -> str:
    """Extract readable text for an element, expanding to the row context.

    A bare inner_text() on a label cell returns just the label (e.g.
    "Savings balance"); the expansion walks up to the enclosing row so the
    value comes along ("Savings balance  $4,321.09"). Falls back to
    inner_text() when the script cannot run.
    """
    try:
        return str(locator.evaluate(_READ_TEXT_JS))
    except Exception:
        return locator.inner_text()
    return locator


class WebSurface:
    """Playwright-backed browser surface with perceive/act primitives."""

    def __init__(self) -> None:
        self._playwright = None
        self.browser = None
        self.context = None
        self.page: Page | None = None

    def start(self, entry_url: str, headless: bool = True, slow_mo: int = 0) -> None:
        """Launch the browser, open a fresh context/page, and go to entry_url."""
        self._playwright = sync_playwright().start()
        self.browser = self._playwright.chromium.launch(headless=headless, slow_mo=slow_mo)
        self.context = self.browser.new_context()
        self.page = self.context.new_page()
        self.page.goto(entry_url)

    def goto(self, url: str) -> None:
        self._require_page().goto(url)

    def current_url(self) -> str:
        return self._require_page().url

    def perceive(self, screenshot: bool = True) -> SurfaceState:
        """Capture the current page as a structured observation."""
        page = self._require_page()
        try:
            snapshot = page.accessibility.snapshot()
        except Exception:
            snapshot = None
        a11y_lines = _render_a11y(snapshot) if snapshot else ["[a11y snapshot unavailable]"]
        try:
            dom = page.content()
        except Exception:
            dom = ""
        png = None
        if screenshot:
            try:
                png = page.screenshot()
            except Exception:
                png = None
        return SurfaceState(
            url=page.url,
            title=page.title(),
            a11y_tree="\n".join(a11y_lines),
            dom_excerpt=dom[:4000],
            screenshot_png=png,
        )

    def act(self, step: dict) -> dict:
        """Execute one action step.

        step keys follow ActionDecision fields: action plus target, value, key,
        ms, url, selector as needed. Returns {"ok", "detail", "strategy_index"}.
        For read, also includes "text" with the extracted content.
        """
        page = self._require_page()
        action = step.get("action")
        try:
            if action == "navigate":
                url = step.get("url") or step.get("value")
                if not url:
                    return {"ok": False, "detail": "navigate requires a url", "strategy_index": None}
                page.goto(url)
                return {"ok": True, "detail": f"Navigated to {url}", "strategy_index": None}

            if action == "wait":
                ms = step.get("ms")
                selector = step.get("selector")
                if ms is not None:
                    page.wait_for_timeout(int(ms))
                    return {"ok": True, "detail": f"Waited {ms} ms", "strategy_index": None}
                if selector:
                    page.wait_for_selector(selector)
                    return {"ok": True, "detail": f"Selector appeared: {selector}", "strategy_index": None}
                return {"ok": False, "detail": "wait requires ms or selector", "strategy_index": None}

            if action == "press":
                key = step.get("key")
                if not key:
                    return {"ok": False, "detail": "press requires a key", "strategy_index": None}
                target = step.get("target")
                if target:
                    locator, index = resolve_strategies(page, build_strategy_chain(target, page))
                    locator.press(key)
                    return {"ok": True, "detail": f"Pressed {key} on target", "strategy_index": index}
                page.keyboard.press(key)
                return {"ok": True, "detail": f"Pressed {key}", "strategy_index": None}

            if action in ("click", "fill", "select", "read"):
                target = step.get("target")
                strategies = build_strategy_chain(target, page)
                locator, index = resolve_strategies(page, strategies)
                if action == "click":
                    locator.click()
                    return {"ok": True, "detail": "Clicked target", "strategy_index": index}
                if action == "fill":
                    value = step.get("value")
                    if value is None:
                        return {"ok": False, "detail": "fill requires a value", "strategy_index": index}
                    locator = _editable_locator(page, locator)
                    locator.fill(value)
                    return {"ok": True, "detail": f"Filled target with {len(value)} chars", "strategy_index": index}
                if action == "select":
                    value = step.get("value")
                    if value is None:
                        return {"ok": False, "detail": "select requires a value", "strategy_index": index}
                    locator = _editable_locator(page, locator)
                    locator.select_option(value)
                    return {"ok": True, "detail": f"Selected option {value!r}", "strategy_index": index}
                if action == "read":
                    try:
                        text = locator.evaluate(_READ_TEXT_JS)
                    except Exception:
                        text = locator.inner_text()
                    return {
                        "ok": True,
                        "detail": f"Read {len(text)} chars",
                        "strategy_index": index,
                        "text": text,
                    }

            return {"ok": False, "detail": f"Unknown action: {action!r}", "strategy_index": None}
        except TargetNotFoundError as exc:
            return {"ok": False, "detail": str(exc), "strategy_index": None}
        except Exception as exc:
            return {"ok": False, "detail": f"{type(exc).__name__}: {exc}", "strategy_index": None}

    def close(self) -> None:
        """Tear down page, context, browser, and the Playwright driver."""
        for handle_name in ("page", "context", "browser"):
            handle = getattr(self, handle_name, None)
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
            setattr(self, handle_name, None)
        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:
                pass
            self._playwright = None

    def _require_page(self) -> Page:
        if self.page is None:
            raise RuntimeError("WebSurface is not started; call start(entry_url) first.")
        return self.page
