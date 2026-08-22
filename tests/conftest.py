"""
Shared test fixtures.

The whole pipeline is a thin orchestration layer over LLM and search calls,
so the tests never hit real APIs - they patch `get_llm`/`web_search` at each
agent module's import site and feed back canned responses. This keeps the
suite fast, free, and runnable in CI with no API keys.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


class FakeMessage(SimpleNamespace):
    """Stand-in for the AIMessage LangChain LLM calls return - agents only
    ever read `.content`."""
    content: str


@pytest.fixture
def fake_llm():
    """A MagicMock whose .invoke(...) always returns the same canned content.
    Configure per-test with `fake_llm.invoke.return_value = FakeMessage(...)`
    or `fake_llm.invoke.side_effect = [...]` for multi-call agents."""
    llm = MagicMock()
    llm.invoke.return_value = FakeMessage(content="stub response")
    return llm


@pytest.fixture
def base_state():
    """A minimal, valid CampaignState to build test states from."""
    return {
        "brief": "Plan a launch campaign for a budget fitness tracking app targeting college students.",
        "max_revisions": 2,
        "revision_count": 0,
        "critique_history": [],
        "agent_log": [],
    }
