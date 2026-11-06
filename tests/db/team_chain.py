"""Alembic heads on the team chain, apart from the 0.21 release's own branch.

The tests of each team revision downgrade from head to the revision before
theirs. Upstream's ``d0a1f879a01d`` branched off before the team chain, so that
downgrade leaves it in place as a second head. It says nothing about the chain.
"""

from __future__ import annotations

import sqlalchemy as sa

from omnigent.db.utils import _get_current_db_heads

RELEASE_BRANCH_HEAD = "d0a1f879a01d"


def team_chain_heads(engine: sa.Engine) -> tuple[str, ...]:
    """Return the database's heads without the release branch's own head."""
    return tuple(head for head in _get_current_db_heads(engine) if head != RELEASE_BRANCH_HEAD)
