"""Regression test for the omnigent.llms <-> omnigent.util.reasoning_effort cycle.

The eager top-level imports in ``omnigent/llms/__init__.py`` created
a circular load when any caller imported ``omnigent.llms.errors``
during the load of ``omnigent.util.reasoning_effort`` (which happens on
every server-routes import via ``server/routes/sessions.py``).

The fix in ``omnigent/llms/__init__.py`` switches to a
``__getattr__`` shim so ``Client`` and ``get_model_context_window``
are resolved lazily on first access. This test guards against
re-introducing the cycle by re-importing the affected modules
in a fresh interpreter and asserting both the
short-form and long-form import paths work.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from textwrap import dedent

from tests.budgets import budget


def _run_fresh(body: str) -> None:
    """Probe a cold import without replacing modules held by collected tests.

    Removing sessions from this process's module cache leaves older references
    behind: facade patches then target a different module from its lazy proxies.
    """
    root = Path(__file__).resolve().parents[2]
    probe = f"import sys\nsys.path.insert(0, {str(root)!r})\n" + dedent(body)
    result = subprocess.run(
        [sys.executable, "-I", "-c", probe],
        capture_output=True,
        text=True,
        timeout=budget(30),
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_sessions_routes_import_does_not_trigger_cycle() -> None:
    """Importing routes must not re-enter reasoning_effort through llms.client."""
    _run_fresh("import omnigent.server.routes.sessions")


def test_short_form_import_still_works() -> None:
    """The short-form imports keep working after the lazy-attribute switch."""
    _run_fresh("""
        from omnigent.llms import Client, get_model_context_window

        assert Client is not None
        assert callable(get_model_context_window)
        """)


def test_module_only_import_does_not_load_client() -> None:
    """Importing llms alone must not eagerly load client.py."""
    _run_fresh("""
        import omnigent.llms

        assert "omnigent.llms.client" not in sys.modules, (
            "omnigent.llms.client was imported eagerly; lazy shim regressed"
        )
        """)


def test_unknown_attribute_raises_attribute_error() -> None:
    """The lazy shim preserves normal AttributeError semantics."""
    _run_fresh("""
        import omnigent.llms as llms_pkg

        try:
            llms_pkg.does_not_exist
        except AttributeError as error:
            assert "does_not_exist" in str(error)
        else:
            raise AssertionError("expected AttributeError")
        """)
