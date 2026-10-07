"""Agent loop for the BankGPT computer-use agent.

Perceive, decide, guardrail-check, act, record. The loop stops on done,
escalation, stuck detection, max steps, or timeout, and produces an AgentRun
artifact-ready record. No em dashes are used anywhere in this file.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from .llm import ActionDecision, ActionRequest, LLMClient
from .surface import WebSurface, build_strategy_chain


class RecordedStep(BaseModel):
    index: int
    action: str
    target_strategies: list[dict] = Field(default_factory=list)
    target_rationale: str = ""
    params: dict = Field(default_factory=dict)
    reasoning: str = ""
    obs_before: dict = Field(default_factory=dict)
    obs_after: dict = Field(default_factory=dict)
    checkpoint: dict | None = None


class AgentRun(BaseModel):
    run_id: str
    goal: str
    entry_point: str
    model_name: str
    steps: list[RecordedStep] = Field(default_factory=list)
    status: Literal["completed", "stuck", "escalated", "timeout", "guardrail_blocked"]
    started_at: str
    ended_at: str | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strategy_rationale(strategies: list[dict]) -> str:
    if not strategies:
        return "No target strategies; the action does not address a page element."
    types = [s.get("type") for s in strategies]
    head = " -> ".join(types)
    return (
        f"Strategy chain ({head}), ordered most robust first: role+name leads "
        "because it survives markup churn, label and placeholder follow as "
        "visible-text affordances, and css/xpath/text are exact-match fallbacks."
    )


def _verdict(result: Any) -> str:
    """Normalize a policy check_action result to allow | blocked | requires_human.

    Accepts a dict ({"verdict": ...}) or an object with .verdict / .blocked /
    .requires_human attributes, since the guardrails module defines the exact
    type. Defaults to allow.
    """
    raw: Any = None
    if isinstance(result, dict):
        raw = result.get("verdict")
    else:
        for attr in ("verdict", "decision"):
            if hasattr(result, attr):
                raw = getattr(result, attr)
                break
        if raw is None:
            if getattr(result, "blocked", False):
                return "blocked"
            if getattr(result, "requires_human", False):
                return "requires_human"
    text = str(raw).lower() if raw is not None else "allow"
    if text in ("blocked", "block", "deny", "denied"):
        return "blocked"
    if text in ("requires_human", "escalate", "human", "review"):
        return "requires_human"
    return "allow"


def _escalate(
    escalation_mgr: Any,
    *,
    run_id: str,
    goal: str,
    reason: str,
    step_index: int,
    observation: dict,
) -> None:
    if escalation_mgr is None:
        return
    escalation_mgr.request_intervention(
        run_id=run_id,
        goal=goal,
        reason=reason,
        step_index=step_index,
        observation=observation,
    )


def run_goal(
    goal: str,
    entry_point: str,
    llm: LLMClient,
    surface: WebSurface,
    *,
    policy: Any = None,
    escalation_mgr: Any = None,
    logger: Any = None,
    max_steps: int = 25,
    timeout_s: int = 600,
) -> AgentRun:
    """Run the perceive/decide/act loop until a terminal status is reached."""
    run_id = uuid.uuid4().hex[:12]
    started_at = _now_iso()
    deadline = time.monotonic() + timeout_s
    model_name = getattr(llm, "model", type(llm).__name__)

    if surface.page is None:
        surface.start(entry_point)

    steps: list[RecordedStep] = []
    history: list[dict] = []

    def finish(final_status: Literal["completed", "stuck", "escalated", "timeout", "guardrail_blocked"]) -> AgentRun:
        return AgentRun(
            run_id=run_id,
            goal=goal,
            entry_point=entry_point,
            model_name=model_name,
            steps=steps,
            status=final_status,
            started_at=started_at,
            ended_at=_now_iso(),
        )

    def _stuck_after(record: RecordedStep) -> str | None:
        """Return a stuck reason if the last steps show no progress."""
        if len(steps) < 3:
            return None
        tail = steps[-3:]
        actions_urls = {(s.action, s.obs_after.get("url"), s.obs_after.get("title")) for s in tail}
        if len(actions_urls) == 1:
            s0 = tail[0]
            return (
                f"3 consecutive steps with identical (action={s0.action!r}, "
                f"url={s0.obs_after.get('url')!r}) and no title change"
            )
        clicks = [s for s in tail if s.action == "click"]
        if len(clicks) == 3:
            import json as _json

            targets = {_json.dumps(s.target_strategies, sort_keys=True) for s in clicks}
            states = {(s.obs_after.get("url"), s.obs_after.get("title")) for s in clicks}
            if len(targets) == 1 and len(states) == 1:
                return "Same target clicked 3 times with no state change"
        return None

    step_index = 0
    while True:
        if time.monotonic() > deadline:
            return finish("timeout")

        obs = surface.perceive(screenshot=False)
        obs_before = {"url": obs.url, "title": obs.title}

        request = ActionRequest(
            goal=goal,
            step_index=step_index,
            history=history,
            observation={
                "url": obs.url,
                "title": obs.title,
                "a11y": obs.a11y_tree,
                "dom_excerpt": obs.dom_excerpt,
            },
        )
        decision: ActionDecision = llm.decide(request)

        strategies = build_strategy_chain(decision.target, surface.page)
        rationale = _strategy_rationale(strategies)

        if decision.action in ("done", "escalate"):
            params: dict = {}
            if decision.action == "done" and decision.summary:
                params["summary"] = decision.summary
            if decision.action == "escalate" and decision.escalate_reason:
                params["escalate_reason"] = decision.escalate_reason
            record = RecordedStep(
                index=step_index,
                action=decision.action,
                target_strategies=strategies,
                target_rationale=rationale,
                params=params,
                reasoning=decision.reasoning,
                obs_before=obs_before,
                obs_after=obs_before,
            )
            steps.append(record)
            if logger is not None:
                logger.log("step", {"run_id": run_id, "step": record.model_dump()})
            if decision.action == "done":
                return finish("completed")
            _escalate(
                escalation_mgr,
                run_id=run_id,
                goal=goal,
                reason=decision.escalate_reason or decision.reasoning or "LLM requested escalation",
                step_index=step_index,
                observation=obs_before,
            )
            return finish("escalated")

        action_dict = decision.model_dump(exclude_none=False)
        if policy is not None:
            verdict = _verdict(policy.check_action(action_dict, obs.url))
            if verdict == "blocked":
                return finish("guardrail_blocked")
            if verdict == "requires_human":
                _escalate(
                    escalation_mgr,
                    run_id=run_id,
                    goal=goal,
                    reason="Policy requires human review for this action",
                    step_index=step_index,
                    observation=obs_before,
                )
                return finish("escalated")

        step_payload = {
            "action": decision.action,
            "target": decision.target,
            "value": decision.value,
            "key": decision.key,
            "ms": decision.ms,
        }
        result = surface.act(step_payload)

        params = {
            "value": decision.value,
            "key": decision.key,
            "ms": decision.ms,
            "act_ok": result.get("ok"),
            "act_detail": result.get("detail"),
            "strategy_index": result.get("strategy_index"),
        }
        if decision.action == "read" and result.get("ok") and "text" in result:
            params["read_text"] = result["text"]

        obs_after_state = surface.perceive(screenshot=False)
        obs_after = {"url": obs_after_state.url, "title": obs_after_state.title}

        record = RecordedStep(
            index=step_index,
            action=decision.action,
            target_strategies=strategies,
            target_rationale=rationale,
            params=params,
            reasoning=decision.reasoning,
            obs_before=obs_before,
            obs_after=obs_after,
        )
        steps.append(record)
        history.append(
            {
                "index": step_index,
                "action": decision.action,
                "target": decision.target,
                "ok": result.get("ok"),
                "detail": result.get("detail"),
            }
        )
        if logger is not None:
            logger.log("step", {"run_id": run_id, "step": record.model_dump()})

        stuck_reason = _stuck_after(record)
        if stuck_reason is not None:
            _escalate(
                escalation_mgr,
                run_id=run_id,
                goal=goal,
                reason=f"Stuck: {stuck_reason}",
                step_index=step_index,
                observation=obs_after,
            )
            return finish("escalated" if escalation_mgr is not None else "stuck")

        step_index += 1
        if step_index >= max_steps:
            return finish("stuck")
