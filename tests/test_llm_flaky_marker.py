"""Exercise real model-rotation retry hooks in an isolated pytest child.

The outer suite can disable all retries. Only this plugin self-test deliberately
fails twice; no LLM or product operation is retried.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

_SCENARIO = """
from __future__ import annotations

import pytest

from tests import _model_pools

# Module-global so state survives across rerunfailures attempts.
_SEEN_MODELS: list[str] = []


@pytest.mark.llm_flaky(reruns=2)
def test_llm_flaky_rotates_model_per_attempt(request: pytest.FixtureRequest) -> None:
    # --force-reruns controls the child budget, so verify marker translation too.
    marker = request.node.get_closest_marker("flaky")
    assert marker is not None and marker.kwargs["reruns"] == 2
    _SEEN_MODELS.append(_model_pools.resolve_model("databricks-claude-sonnet-4-6", spread=False))
    if len(_SEEN_MODELS) < 3:
        # Force a rerun; fails loudly if llm_flaky -> flaky is broken.
        raise AssertionError(f"forcing rerun, attempt {len(_SEEN_MODELS)} of 3")
    # 2-member anthropic chain: 3 attempts rotate sonnet -> opus ->
    # sonnet. A constant list means rotation never happened.
    assert _SEEN_MODELS == [
        "databricks-claude-sonnet-4-6",
        "databricks-claude-opus-4-6",
        "databricks-claude-sonnet-4-6",
    ]
"""


def test_llm_flaky_rotates_model_per_attempt(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The marker translates to two retries with per-attempt model rotation."""
    root = str(Path(__file__).resolve().parents[1])
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([root, os.environ.get("PYTHONPATH", "")]))
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    pytester.makeini("[pytest]\nmarkers = llm_flaky: model-rotation retry self-test\n")
    pytester.makeconftest("""
from tests.conftest import pytest_collection_modifyitems, pytest_runtest_setup
""")
    pytester.makepyfile(_SCENARIO)
    # Explicitly scoped to the child: parent --force-reruns 0 remains untouched.
    result = pytester.runpytest_subprocess(
        "-q", "-p", "pytest_rerunfailures", "--force-reruns", "2"
    )
    result.assert_outcomes(passed=1)
    assert result.parseoutcomes()["rerun"] == 2
