"""BankGPT computer-use agent: LLM client, web surface, and agent loop."""

from .agent import AgentRun, RecordedStep, run_goal
from .llm import (
    ActionDecision,
    ActionRequest,
    LLMClient,
    OpenAICompatClient,
    ScriptedMockClient,
    default_demo_script,
)
from .surface import (
    SurfaceState,
    TargetNotFoundError,
    WebSurface,
    build_strategy_chain,
    resolve_strategies,
)

__all__ = [
    "ActionDecision",
    "ActionRequest",
    "AgentRun",
    "LLMClient",
    "OpenAICompatClient",
    "RecordedStep",
    "ScriptedMockClient",
    "SurfaceState",
    "TargetNotFoundError",
    "WebSurface",
    "build_strategy_chain",
    "default_demo_script",
    "resolve_strategies",
    "run_goal",
]
