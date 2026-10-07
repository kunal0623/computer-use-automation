"""Deterministic replay engine for capability artifacts.

Replays a CapabilityArtifact against a web surface with no LLM calls:
fixed strategy order, explicit waits only, no randomness, no
wall-clock-dependent branching.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal, TYPE_CHECKING

from pydantic import BaseModel, Field

from bankgpt_cua.artifact import (
    ArtifactStep,
    CapabilityArtifact,
    Checkpoint,
    LocatorStrategy,
)
from bankgpt_cua import errors as error_taxonomy
from bankgpt_cua.errors import OutcomeClass

if TYPE_CHECKING:  # pragma: no cover
    from bankgpt_cua.agent import AgentRun

try:  # Resolution helper built by the surface worker; fall back locally.
    from bankgpt_cua.surface import (  # type: ignore
        resolve_strategies as _surface_resolve_strategies,
        editable_locator as _surface_editable_locator,
        readable_text as _surface_readable_text,
        TargetNotFoundError,
    )
except Exception:  # pragma: no cover

    class TargetNotFoundError(Exception):
        """Raised when no locator strategy in a chain resolves."""


    _surface_resolve_strategies = None  # type: ignore

    def _surface_editable_locator(page: Any, locator: Any) -> Any:  # type: ignore
        return locator

    def _surface_readable_text(page: Any, locator: Any) -> str:  # type: ignore
        try:
            return str(locator.inner_text())
        except Exception:
            return ""


class ReplayResult(BaseModel):
    """Outcome of one deterministic replay."""

    run_id: str
    artifact_id: str
    artifact_version: str
    status: Literal["success", "business_outcome", "recovered", "hard_failure"]
    outputs: dict[str, Any] = Field(default_factory=dict)
    steps_completed: int = 0
    recoveries_applied: list[str] = Field(default_factory=list)
    failure: dict[str, Any] | None = None  # {step_index, expected, observed}
    business_outcome: dict[str, Any] | None = None  # {code, message}
    evidence_dir: str | None = None


_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def substitute(value: Any, context: dict[str, Any]) -> Any:
    """Recursively replace ${name} placeholders in strings."""
    if isinstance(value, str):
        def _rep(m: re.Match) -> str:
            key = m.group(1)
            return str(context[key]) if key in context else m.group(0)

        return _PLACEHOLDER_RE.sub(_rep, value)
    if isinstance(value, dict):
        return {k: substitute(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [substitute(v, context) for v in value]
    return value


def _coerce_input(name: str, declared: str, raw: Any) -> Any:
    """Coerce a raw input value to the declared type, or raise ValueError."""
    if declared == "string":
        if isinstance(raw, bool):
            raise ValueError(f"input '{name}' must be a string, got bool")
        if isinstance(raw, (int, float)):
            return str(raw)
        if not isinstance(raw, str):
            raise ValueError(f"input '{name}' must be a string, got {type(raw).__name__}")
        return raw
    if declared == "integer":
        if isinstance(raw, bool):
            raise ValueError(f"input '{name}' must be an integer, got bool")
        if isinstance(raw, int):
            return raw
        if isinstance(raw, str) and re.fullmatch(r"[+-]?\d+", raw.strip()):
            return int(raw.strip())
        raise ValueError(f"input '{name}' must be an integer, got {raw!r}")
    if declared == "number":
        if isinstance(raw, bool):
            raise ValueError(f"input '{name}' must be a number, got bool")
        if isinstance(raw, (int, float)):
            return float(raw)
        if isinstance(raw, str):
            try:
                return float(raw.strip())
            except ValueError:
                pass
        raise ValueError(f"input '{name}' must be a number, got {raw!r}")
    if declared == "boolean":
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            lowered = raw.strip().lower()
            if lowered in ("true", "1", "yes", "y"):
                return True
            if lowered in ("false", "0", "no", "n"):
                return False
        raise ValueError(f"input '{name}' must be a boolean, got {raw!r}")
    raise ValueError(f"input '{name}' has unknown declared type {declared!r}")


def _validate_inputs(
    artifact: CapabilityArtifact, inputs: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Validate and coerce inputs. Returns (coerced, violations)."""
    coerced: dict[str, Any] = {}
    violations: list[str] = []
    inputs = inputs or {}
    for param in artifact.inputs:
        if param.name not in inputs or inputs[param.name] is None:
            if param.required:
                violations.append(f"missing required input '{param.name}'")
            continue
        raw = inputs[param.name]
        try:
            value = _coerce_input(param.name, param.type, raw)
        except ValueError as exc:
            violations.append(str(exc))
            continue
        if param.pattern and param.type == "string":
            if not re.fullmatch(param.pattern, value):
                violations.append(
                    f"input '{param.name}' does not match pattern {param.pattern!r}"
                )
                continue
        coerced[param.name] = value
    return coerced, violations


def _fallback_resolve_strategies(page: Any, strategies: list[dict[str, Any]]) -> tuple[Any, int]:
    """Local strategy resolver used when bankgpt_cua.surface is unavailable.

    Implements the documented strategy dict contract:
    {"type": "css"|"xpath"|"text"|"role"|"placeholder"|"label"|"attr"|"image", ...}.
    Returns (locator, winning_index); raises TargetNotFoundError.
    """
    for i, s in enumerate(strategies):
        stype = s.get("type")
        try:
            if stype == "css":
                locator = page.locator(s["value"])
            elif stype == "xpath":
                locator = page.locator(f"xpath={s['value']}")
            elif stype == "text":
                locator = page.get_by_text(s["value"])
            elif stype == "role":
                locator = page.get_by_role(s["role"], name=s.get("name"))
            elif stype == "placeholder":
                locator = page.get_by_placeholder(s["value"])
            elif stype == "label":
                locator = page.get_by_label(s["value"])
            elif stype == "attr":
                locator = page.locator(f"[{s['attr_name']}='{s['attr_value']}']")
            elif stype == "image":
                locator = page.get_by_role("img", name=s.get("value") or s.get("name"))
            else:
                continue
            if locator.count() > 0:
                return locator, i
        except TargetNotFoundError:
            raise
        except Exception:
            continue
    raise TargetNotFoundError(
        f"No strategy resolved from chain of {len(strategies)} strategies."
    )


def _make_resolver(surface: Any):
    """Pick a resolve(page, strategies) callable from the surface or fallbacks."""
    candidate = getattr(surface, "resolve_strategies", None)
    if candidate is None and _surface_resolve_strategies is not None:
        candidate = _surface_resolve_strategies
    if candidate is None:
        candidate = _fallback_resolve_strategies

    def _resolve(page: Any, strategies: list[dict[str, Any]]) -> tuple[Any, int]:
        try:
            return candidate(page, strategies)
        except TypeError:
            return candidate(strategies)  # bound method without page arg

    return _resolve


def _strategy_dicts(target: Any) -> list[dict[str, Any]]:
    """Normalize a LocatorTarget (or dict) to a list of strategy dicts."""
    if target is None:
        return []
    if isinstance(target, dict):
        strategies = target.get("strategies", [])
        out: list[dict[str, Any]] = []
        for s in strategies:
            out.append(dict(s) if isinstance(s, dict) else s.model_dump(exclude_none=True))
        return out
    strategies = getattr(target, "strategies", []) or []
    out2: list[dict[str, Any]] = []
    for s in strategies:
        out2.append(dict(s) if isinstance(s, dict) else s.model_dump(exclude_none=True))
    return out2


def _redacted_params(
    params: dict[str, Any],
    artifact: CapabilityArtifact,
    values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Redact input values flagged redact_in_logs from logged params.

    Step params use generic keys ("value"), so redaction is by value: any
    string containing the runtime value of a redacted input is masked.
    """
    secrets: set[str] = set()
    for p in artifact.inputs:
        if p.redact_in_logs and values and values.get(p.name) is not None:
            secrets.add(str(values[p.name]))

    def _mask(value: Any) -> Any:
        if isinstance(value, str):
            for secret in secrets:
                if secret and secret in value:
                    value = value.replace(secret, "***")
            return value
        if isinstance(value, dict):
            return {k: _mask(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_mask(v) for v in value]
        return value

    return _mask(params)


def _log(logger: Any, event: str, payload: dict[str, Any]) -> None:
    if logger is None:
        return
    try:
        logger.log(event, payload)
    except Exception:
        pass


def _perceive(page: Any, dialog_texts: list[str]) -> tuple[str, str]:
    """Return (url, text) where text is title + body text + dialog texts."""
    try:
        url = page.url
    except Exception:
        url = ""
    try:
        title = page.title()
    except Exception:
        title = ""
    try:
        body_text = page.evaluate("() => document.body ? document.body.innerText : ''")
    except Exception:
        body_text = ""
    parts = [title or "", body_text or ""]
    parts.extend(dialog_texts)
    text = "\n".join(p for p in parts if p)
    return url, text


def _postprocess(text: str, mode: str) -> str:
    if mode == "strip_currency":
        return text.replace("$", "").replace(",", "").strip()
    if mode == "extract_currency":
        match = re.search(r"\$?([\d,]+\.\d{2})", text)
        if not match:
            raise ValueError(f"no currency amount found in extracted text: {text!r}")
        return match.group(1).replace(",", "")
    if mode == "strip_whitespace":
        return text.strip()
    return text


def _find_policy_entry(
    artifact: CapabilityArtifact, name: str
) -> Any | None:
    for entry in artifact.error_policy:
        if entry.name == name:
            return entry
    return None


class _Replayer:
    """Stateful single-replay driver (sync Playwright)."""

    def __init__(
        self,
        artifact: CapabilityArtifact,
        coerced_inputs: dict[str, Any],
        surface: Any,
        policy: Any,
        escalation_mgr: Any,
        logger: Any,
        headless: bool,
        max_recoveries: int,
    ) -> None:
        self.artifact = artifact
        self.inputs = coerced_inputs
        self.surface = surface
        self.policy = policy
        self.escalation_mgr = escalation_mgr
        self.logger = logger
        self.headless = headless
        self.max_recoveries = max_recoveries
        self.context = {
            "entry_point": (artifact.surface or {}).get("entry_point", ""),
            **coerced_inputs,
        }
        self.resolve = _make_resolver(surface)
        self.dialog_texts: list[str] = []
        self.outputs: dict[str, Any] = {}
        self.recoveries: list[str] = []
        self.steps_completed = 0
        self.run_id = self._deterministic_run_id()
        self.page: Any = None

    def _deterministic_run_id(self) -> str:
        canonical = json.dumps(
            {"artifact": self.artifact.id, "version": self.artifact.version, "inputs": self.inputs},
            sort_keys=True,
            default=str,
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        return f"{self.artifact.id}-{digest}"

    # -- lifecycle ----------------------------------------------------

    def run(self) -> ReplayResult:
        entry = self.context["entry_point"]
        _log(self.logger, "replay_start", {"run_id": self.run_id, "artifact": self.artifact.id})
        try:
            self.page = self._start_surface(entry)
        except Exception as exc:
            return self._hard_failure(
                step_index=-1,
                expected=f"surface start at {entry}",
                observed=f"{type(exc).__name__}: {exc}",
                code="SURFACE_START_FAILED",
            )
        try:
            self.page.on("dialog", self._on_dialog)
        except Exception:
            pass
        try:
            result = self._run_steps()
        finally:
            close = getattr(self.surface, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        return result

    def _start_surface(self, entry: str) -> Any:
        try:
            started = self.surface.start(entry, headless=self.headless)
        except TypeError:
            started = self.surface.start(entry)
        # start() may return None and expose the page as an attribute.
        return (
            getattr(started, "page", None)
            or getattr(self.surface, "page", None)
            or started
        )

    def _on_dialog(self, dialog: Any) -> None:
        try:
            self.dialog_texts.append(str(dialog.message))
        except Exception:
            self.dialog_texts.append("<unreadable dialog>")
        try:
            dialog.dismiss()
        except Exception:
            pass

    # -- step execution -----------------------------------------------

    def _run_steps(self) -> ReplayResult:
        steps = sorted(self.artifact.steps, key=lambda s: s.index)
        for step in steps:
            if step.action == "done":
                self.steps_completed += 1
                _log(self.logger, "step_done", {"index": step.index, "action": "done"})
                break
            url, _ = _perceive(self.page, self.dialog_texts)
            if self.policy is not None:
                denial = self._guardrail_check(step, url)
                if denial is not None:
                    return denial
            params = substitute(step.params or {}, self.context)
            _log(
                self.logger,
                "step_start",
                {
                    "index": step.index,
                    "action": step.action,
                    "params": _redacted_params(params, self.artifact, self.inputs),
                },
            )
            try:
                self._do_action(step, params)
            except TargetNotFoundError as exc:
                return self._hard_failure(
                    step_index=step.index,
                    expected="locator strategy chain to resolve",
                    observed=str(exc),
                    code="TARGET_NOT_FOUND",
                )
            except Exception as exc:
                return self._hard_failure(
                    step_index=step.index,
                    expected=f"action {step.action} to complete",
                    observed=f"{type(exc).__name__}: {exc}",
                    code="ACTION_FAILED",
                )
            self.steps_completed += 1
            _log(self.logger, "step_done", {"index": step.index, "action": step.action})

            outcome = self._check_errors()
            if outcome is not None:
                return outcome

            miss = self._verify_checkpoint(step.checkpoint)
            if miss is not None:
                try:
                    self.page.wait_for_timeout(1000)
                except Exception:
                    pass
                miss = self._verify_checkpoint(step.checkpoint)
                if miss is not None:
                    expected, observed = miss
                    return self._hard_failure(
                        step_index=step.index,
                        expected=expected,
                        observed=observed,
                        code="CHECKPOINT_MISS",
                    )
            _log(self.logger, "checkpoint_ok", {"index": step.index})

        cond_miss = self._verify_checkpoint(self.artifact.success_condition)
        if cond_miss is not None:
            expected, observed = cond_miss
            return self._hard_failure(
                step_index=-1,
                expected=f"success_condition: {expected}",
                observed=observed,
                code="SUCCESS_CONDITION_MISS",
            )

        self._extract_outputs()
        status: Literal["success", "recovered"] = (
            "recovered" if self.recoveries else "success"
        )
        _log(
            self.logger,
            "replay_end",
            {"run_id": self.run_id, "status": status, "outputs": list(self.outputs)},
        )
        return ReplayResult(
            run_id=self.run_id,
            artifact_id=self.artifact.id,
            artifact_version=self.artifact.version,
            status=status,
            outputs=self.outputs,
            steps_completed=self.steps_completed,
            recoveries_applied=list(self.recoveries),
            evidence_dir=getattr(self.logger, "evidence_dir", None),
        )

    def _guardrail_check(self, step: ArtifactStep, url: str) -> ReplayResult | None:
        params = substitute(step.params or {}, self.context)
        check_url = url
        if step.action == "navigate":
            # Evaluate the destination against the allowlist, not the page
            # we are leaving.
            check_url = str(params.get("url") or params.get("value") or url)
        try:
            action_dict = {
                "action": step.action,
                "target": (
                    step.target.model_dump(exclude_none=True) if step.target else None
                ),
                "params": _redacted_params(params, self.artifact, self.inputs),
            }
            if step.action == "navigate":
                # The policy reads the destination from action["value"].
                action_dict["value"] = params.get("url") or params.get("value")
            verdict = self.policy.check_action(action_dict, check_url)
        except Exception as exc:
            return self._hard_failure(
                step_index=step.index,
                expected="guardrail check to complete",
                observed=f"{type(exc).__name__}: {exc}",
                code="GUARDRAIL_ERROR",
            )
        if isinstance(verdict, dict) and not verdict.get("allowed", True):
            reason = verdict.get("reason", "denied by guardrail policy")
            _log(self.logger, "guardrail_denied", {"index": step.index, "reason": reason})
            return self._hard_failure(
                step_index=step.index,
                expected="guardrail to allow action",
                observed=reason,
                code="GUARDRAIL_DENIED",
            )
        return None

    def _do_action(self, step: ArtifactStep, params: dict[str, Any]) -> None:
        action = step.action
        if action == "navigate":
            url = params.get("url") or params.get("value") or ""
            self.page.goto(url, wait_until="load")
        elif action == "wait":
            self.page.wait_for_timeout(int(params.get("ms", 1000)))
        elif action == "read":
            strategies = _strategy_dicts(step.target)
            locator, win = self.resolve(self.page, strategies)
            _log(
                self.logger,
                "strategy_resolved",
                {"index": step.index, "winning_strategy": win, "action": "read"},
            )
            text = _surface_readable_text(self.page, locator)
            name = params.get("name", params.get("output"))
            if name:
                # A read step explicitly names its output slot; declared
                # OutputDefs are extracted separately by _extract_outputs.
                self.outputs[str(name)] = _postprocess(
                    text, str(params.get("postprocess", "none"))
                )
            _log(
                self.logger,
                "read",
                {"index": step.index, "chars": len(text), "named": bool(name)},
            )
        else:
            strategies = _strategy_dicts(step.target)
            if action == "press" and not strategies:
                self.page.keyboard.press(str(params.get("key", "Enter")))
                return
            locator, win = self.resolve(self.page, strategies)
            _log(
                self.logger,
                "strategy_resolved",
                {"index": step.index, "winning_strategy": win, "action": action},
            )
            if action == "click":
                locator.click()
            elif action == "fill":
                locator = _surface_editable_locator(self.page, locator)
                locator.fill(str(params.get("value", "")))
            elif action == "press":
                locator.press(str(params.get("key", "Enter")))
            elif action == "select":
                locator = _surface_editable_locator(self.page, locator)
                locator.select_option(params.get("value"))
            else:
                raise ValueError(f"unsupported action {action!r}")

    # -- error classification and recovery -----------------------------

    def _check_errors(self) -> ReplayResult | None:
        """Classify current state; apply recoveries; return a terminal result or None."""
        while True:
            url, text = _perceive(self.page, self.dialog_texts)
            matches = error_taxonomy.classify(text, url)
            if not matches:
                return None
            ordered = sorted(
                matches,
                key=lambda m: {
                    OutcomeClass.HARD_FAILURE: 0,
                    OutcomeClass.BUSINESS_OUTCOME: 1,
                    OutcomeClass.RECOVERABLE: 2,
                }[m.outcome_class],
            )
            match = ordered[0]
            _log(
                self.logger,
                "error_matched",
                {"name": match.name, "outcome_class": match.outcome_class.value},
            )
            if match.outcome_class == OutcomeClass.BUSINESS_OUTCOME:
                return self._business_outcome(match)
            if match.outcome_class == OutcomeClass.HARD_FAILURE:
                return self._hard_failure(
                    step_index=self.steps_completed - 1,
                    expected="no hard application error",
                    observed=match.evidence.get("text_excerpt", ""),
                    code=f"APP_ERROR_{match.name.upper()}",
                )
            # RECOVERABLE
            entry = _find_policy_entry(self.artifact, match.name)
            recovery = (entry.recovery if entry else None) or match.recovery
            if recovery is None:
                return self._hard_failure(
                    step_index=self.steps_completed - 1,
                    expected=f"recovery for {match.name}",
                    observed="no recovery action defined",
                    code="RECOVERY_UNDEFINED",
                )
            if len(self.recoveries) >= self.max_recoveries:
                return self._hard_failure(
                    step_index=self.steps_completed - 1,
                    expected=f"recover within {self.max_recoveries} recoveries",
                    observed=f"still failing on {match.name}",
                    code="RECOVERY_EXHAUSTED",
                )
            self._apply_recovery(recovery)
            self.recoveries.append(match.name)
            _log(
                self.logger,
                "recovery_applied",
                {"name": match.name, "count": len(self.recoveries)},
            )

    def _apply_recovery(self, recovery: dict[str, Any]) -> None:
        action = recovery.get("action")
        params = substitute(recovery.get("params") or {}, self.context)
        if action == "navigate":
            self.page.goto(params.get("url", ""), wait_until="load")
        elif action == "wait":
            self.page.wait_for_timeout(int(params.get("ms", 1000)))
        elif action == "click":
            strategies = _strategy_dicts(recovery.get("target"))
            locator, _win = self.resolve(self.page, strategies)
            locator.click()
        else:
            raise ValueError(f"unsupported recovery action {action!r}")

    def _business_outcome(self, match: Any) -> ReplayResult:
        entry = _find_policy_entry(self.artifact, match.name)
        if entry is not None:
            code = (entry.outcome or {}).get("code", match.name.upper())
            template = (entry.outcome or {}).get("message_template", "")
        else:
            code = match.name.upper()
            template = match.evidence.get("text_excerpt", "")
        try:
            message = template.format(**{k: str(v) for k, v in self.context.items()})
        except Exception:
            message = template
        _log(self.logger, "business_outcome", {"code": code})
        return ReplayResult(
            run_id=self.run_id,
            artifact_id=self.artifact.id,
            artifact_version=self.artifact.version,
            status="business_outcome",
            outputs=dict(self.outputs),
            steps_completed=self.steps_completed,
            recoveries_applied=list(self.recoveries),
            business_outcome={"code": code, "message": message},
            evidence_dir=getattr(self.logger, "evidence_dir", None),
        )

    # -- checkpoints ----------------------------------------------------

    def _verify_checkpoint(self, checkpoint: Checkpoint | None) -> tuple[str, str] | None:
        """Return (expected, observed) on miss, else None."""
        if checkpoint is None:
            return None
        # Checkpoint values may reference inputs (${member_id}); resolve them.
        value = substitute(checkpoint.value, self.context)
        url, text = _perceive(self.page, self.dialog_texts)
        if checkpoint.type == "url_contains":
            if value in url:
                return None
            return (
                f"url contains {value!r} ({checkpoint.description})",
                f"url was {url!r}",
            )
        if checkpoint.type == "text_present":
            if value in text:
                return None
            excerpt = text[:200].replace("\n", " ")
            return (
                f"text present {value!r} ({checkpoint.description})",
                f"page text started with {excerpt!r}",
            )
        if checkpoint.type == "element_visible":
            try:
                strategy = LocatorStrategy(
                    type="text", value=checkpoint.value
                ).model_dump(exclude_none=True)
            except Exception:
                strategy = {"type": "text", "value": checkpoint.value}
            try:
                locator, _win = self.resolve(self.page, [strategy])
                if locator.is_visible():
                    return None
                return (
                    f"element visible for {checkpoint.value!r} ({checkpoint.description})",
                    "element resolved but not visible",
                )
            except TargetNotFoundError:
                return (
                    f"element visible for {checkpoint.value!r} ({checkpoint.description})",
                    "no strategy resolved to an element",
                )
        return (f"unknown checkpoint type {checkpoint.type!r}", "")

    # -- outputs ---------------------------------------------------------

    def _extract_outputs(self) -> None:
        for out in self.artifact.outputs:
            ex = out.extraction or {}
            strategy = ex.get("strategy") or {}
            if isinstance(strategy, dict):
                strategy_dict = dict(strategy)
            else:
                strategy_dict = strategy.model_dump(exclude_none=True)
            postprocess = str(ex.get("postprocess", "none"))
            try:
                locator, _win = self.resolve(self.page, [strategy_dict])
                text = _surface_readable_text(self.page, locator)
            except Exception as exc:
                raise RuntimeError(
                    f"output extraction failed for {out.name!r}: {exc}"
                ) from exc
            value = _postprocess(text, postprocess)
            self.outputs[out.name] = value
            _log(
                self.logger,
                "output_extracted",
                {"name": out.name, "postprocess": postprocess},
            )

    # -- terminal failures ------------------------------------------------

    def _hard_failure(
        self, *, step_index: int, expected: str, observed: str, code: str
    ) -> ReplayResult:
        screenshot: bytes | None = None
        if self.logger is not None and self.page is not None:
            try:
                screenshot = self.page.screenshot()
                self.logger.screenshot("failure", screenshot)
            except Exception:
                screenshot = None
            try:
                self.logger.dom("failure", self.page.content())
            except Exception:
                pass
        _log(
            self.logger,
            "hard_failure",
            {"code": code, "step_index": step_index, "expected": expected},
        )
        if self.escalation_mgr is not None:
            try:
                url, text = _perceive(self.page, self.dialog_texts)
            except Exception:
                url, text = "", ""
            try:
                self.escalation_mgr.request_intervention(
                    self.run_id,
                    (self.artifact.provenance or {}).get("goal") or self.artifact.name,
                    step_index,
                    f"{code}: {expected}. Observed: {observed}",
                    f"url={url} text={text[:500]}",
                    screenshot,
                )
            except Exception:
                pass
        return ReplayResult(
            run_id=self.run_id,
            artifact_id=self.artifact.id,
            artifact_version=self.artifact.version,
            status="hard_failure",
            outputs=dict(self.outputs),
            steps_completed=self.steps_completed,
            recoveries_applied=list(self.recoveries),
            failure={
                "step_index": step_index,
                "expected": expected,
                "observed": observed,
                "code": code,
            },
            evidence_dir=getattr(self.logger, "evidence_dir", None),
        )


def replay(
    artifact: CapabilityArtifact,
    inputs: dict[str, Any],
    surface: Any,
    *,
    policy: Any = None,
    escalation_mgr: Any = None,
    logger: Any = None,
    headless: bool = True,
    max_recoveries: int = 3,
) -> ReplayResult:
    """Replay a capability artifact deterministically. No LLM calls.

    Input validation happens before any browser work: violations return a
    business_outcome result with code INVALID_INPUT.
    """
    coerced, violations = _validate_inputs(artifact, inputs or {})
    if violations:
        digest = hashlib.sha256(
            json.dumps(
                {"artifact": artifact.id, "version": artifact.version},
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:16]
        return ReplayResult(
            run_id=f"{artifact.id}-{digest}",
            artifact_id=artifact.id,
            artifact_version=artifact.version,
            status="business_outcome",
            business_outcome={
                "code": "INVALID_INPUT",
                "message": "; ".join(violations),
            },
        )
    driver = _Replayer(
        artifact,
        coerced,
        surface,
        policy=policy,
        escalation_mgr=escalation_mgr,
        logger=logger,
        headless=headless,
        max_recoveries=max_recoveries,
    )
    return driver.run()
