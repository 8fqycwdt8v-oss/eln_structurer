"""Stub for the Anthropic Agent SDK transport.

Lets tests drive ``agent.extract`` end-to-end without making a network
call. We intercept the SDK's ``query`` and ``ClaudeSDKClient`` so the
agent sees a canned conversation: a primary-loop response, optionally
followed by a critic response.

Usage:

    from tests._sdk_stub import stub_sdk
    with stub_sdk(primary_text="(stubbed assistant response)",
                  critic_text='{"findings": []}') as log:
        result = await extract("paragraph")
        assert log.primary_calls == 1
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import AsyncIterator

# We don't import claude_agent_sdk lazily — tests that use this stub
# already depend on it being installed, and importing here gives us
# the type names for synthetic messages.
from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
)


@dataclass
class StubCallLog:
    """Records what the stubbed SDK was asked to do."""
    primary_calls: int = 0
    critic_calls: int = 0
    last_user_prompts: list[str] = field(default_factory=list)


def _make_result_message(num_turns: int = 1) -> ResultMessage:
    """Construct a synthetic ResultMessage carrying token + cost fields."""
    return ResultMessage(
        subtype="end_of_turn",
        duration_ms=10,
        duration_api_ms=5,
        is_error=False,
        num_turns=num_turns,
        session_id="stub",
        stop_reason="end_turn",
        total_cost_usd=0.0,
        usage={"input_tokens": 100, "output_tokens": 50},
        result=None,
        structured_output=None,
        model_usage=None,
        permission_denials=None,
        deferred_tool_use=None,
        errors=None,
        api_error_status=None,
        uuid="stub-uuid",
    )


@contextlib.contextmanager
def stub_sdk(
    *,
    primary_text: str = "stub primary response",
    critic_text: str | None = '{"findings": []}',
):
    """Patch claude_agent_sdk.query and ClaudeSDKClient.

    primary_text is yielded by ClaudeSDKClient (the agent.extract path).
    critic_text is yielded by query() (the critic.run_critic path); set
    to None to make the critic stub raise or skip.

    Real tool calls don't execute — the stubbed primary just yields a
    text block and a result message. This is fine for tests that check
    the OUTER agent.extract behaviour (size cap, cost accounting,
    revision rollback). For inside-the-loop behaviour, drive the rule
    pack and finalize_reaction handlers directly (see test_loop_state.py).
    """
    import claude_agent_sdk as sdk
    log = StubCallLog()

    class _StubClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def query(self, prompt: str):
            log.primary_calls += 1
            log.last_user_prompts.append(prompt)

        async def receive_response(self) -> AsyncIterator:
            yield AssistantMessage(
                content=[TextBlock(text=primary_text)],
                model="stub",
                parent_tool_use_id=None,
            )
            yield _make_result_message()

    async def _stub_query(*, prompt: str, options=None, transport=None):
        log.critic_calls += 1
        text = critic_text if critic_text is not None else '{"findings": []}'
        yield AssistantMessage(
            content=[TextBlock(text=text)],
            model="stub",
            parent_tool_use_id=None,
        )
        yield _make_result_message()

    original_client = sdk.ClaudeSDKClient
    original_query = sdk.query
    sdk.ClaudeSDKClient = _StubClient  # type: ignore[misc]
    sdk.query = _stub_query  # type: ignore[misc]
    # Mirror onto the agent module's already-imported references.
    import eln_structurer.agent as agent_mod
    agent_mod.ClaudeSDKClient = _StubClient  # type: ignore[misc]
    import eln_structurer.critic.runner as critic_runner
    critic_runner.query = _stub_query  # type: ignore[misc]
    # Predict-side agent (Tier 6) imports the same names locally; patch
    # its module attributes too so the stub catches every call site.
    try:
        import eln_structurer.predict.agent as predict_agent_mod
    except ImportError:
        predict_agent_mod = None  # type: ignore[assignment]
    if predict_agent_mod is not None:
        predict_agent_mod.ClaudeSDKClient = _StubClient  # type: ignore[misc]
    try:
        yield log
    finally:
        sdk.ClaudeSDKClient = original_client  # type: ignore[misc]
        sdk.query = original_query  # type: ignore[misc]
        agent_mod.ClaudeSDKClient = original_client  # type: ignore[misc]
        critic_runner.query = original_query  # type: ignore[misc]
        if predict_agent_mod is not None:
            predict_agent_mod.ClaudeSDKClient = original_client  # type: ignore[misc]
