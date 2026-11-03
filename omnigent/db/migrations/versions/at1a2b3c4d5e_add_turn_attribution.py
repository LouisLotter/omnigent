"""Record who did what on whose behalf, with each turn and approval.

Revision ID: at1a2b3c4d5e
Revises: ap1a2b3c4d5e
Create Date: 2026-11-03 00:00:00.000000

Adds:

- ``conversation_items.attribution``: the JSON attribution of the turn an item
  starts (initiator, thread owner, computer, harness login, harness, model and
  effort), written in the same insert as the item. ``NULL`` for every other
  item and for items saved before this revision.
- ``approval_verdicts.attribution``: who asked for an approval, written in the
  same insert as the ledger row.
- ``ix_approval_verdicts_session``: lists a thread's approvals and deletes them
  with the thread.
- ``session_inputs``: every input accepted into a thread and the person
  answerable for it, written before the input reaches a runner.
- ``session_turns``: each turn a runner registered, with the inputs it starts
  with. Approvals read who took part in a turn from these two tables.

Deployment: additive. Older application code ignores the columns and index;
no existing row changes. Each step runs only when missing, so an upgrade
interrupted after MySQL or CockroachDB committed part of it resumes on the next
start. A conversation database hosted separately is created by
``create_all`` and never migrated: on one created before this revision the
item column is absent, and the application reads attribution there as unknown.
Downgrading drops both tables, both columns and the index, and the
attribution in them.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from omnigent.db.db_models import Uuid16

revision: str = "at1a2b3c4d5e"
down_revision: str | None = "ap1a2b3c4d5e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_INDEX = "ix_approval_verdicts_session"
_TABLES = ("session_inputs", "session_turns")
_INDEX_COLUMNS = ("workspace_id", "session_id", "created_at", "elicitation_id")


def _columns(bind: sa.Connection, table: str) -> set[str]:
    return {column["name"] for column in sa.inspect(bind).get_columns(table)}


def _has_index(bind: sa.Connection) -> bool:
    return _INDEX in {index["name"] for index in sa.inspect(bind).get_indexes("approval_verdicts")}


def _create_tables(bind: sa.Connection) -> None:
    """Create each missing table, then each missing index."""
    inspector = sa.inspect(bind)
    if not inspector.has_table("session_inputs"):
        op.create_table(
            "session_inputs",
            sa.Column("workspace_id", sa.BigInteger(), nullable=False),
            sa.Column("session_id", Uuid16(), nullable=False),
            sa.Column("input_id", sa.String(160), nullable=False),
            sa.Column("kind", sa.SmallInteger(), nullable=False),
            sa.Column("person", sa.String(128), nullable=True),
            sa.Column("possible", sa.Text(), nullable=True),
            sa.Column("item_id", sa.String(64), nullable=True),
            sa.Column("text_digest", sa.String(64), nullable=True),
            sa.Column("accepted_at", sa.BigInteger(), nullable=False),
            sa.PrimaryKeyConstraint("workspace_id", "session_id", "input_id"),
            sa.CheckConstraint("kind IN (1, 2, 3, 4, 5)", name="ck_session_inputs_kind"),
        )
    if not inspector.has_table("session_turns"):
        op.create_table(
            "session_turns",
            sa.Column("workspace_id", sa.BigInteger(), nullable=False),
            sa.Column("session_id", Uuid16(), nullable=False),
            sa.Column("turn_id", sa.String(128), nullable=False),
            sa.Column("runner_id", sa.String(64), nullable=True),
            sa.Column("initiator", sa.String(128), nullable=True),
            sa.Column("inputs", sa.Text(), nullable=False),
            sa.Column("unresolved", sa.Boolean(), nullable=False),
            sa.Column("started_at", sa.BigInteger(), nullable=False),
            sa.PrimaryKeyConstraint("workspace_id", "session_id", "turn_id"),
        )
    for table, name, columns in (
        (
            "session_inputs",
            "ix_session_inputs_accepted",
            ("workspace_id", "session_id", "accepted_at"),
        ),
        (
            "session_turns",
            "ix_session_turns_started",
            ("workspace_id", "session_id", "started_at"),
        ),
    ):
        if name not in {index["name"] for index in sa.inspect(bind).get_indexes(table)}:
            op.create_index(name, table, list(columns))


def upgrade() -> None:
    """Add each missing table, column and index."""
    bind = op.get_bind()
    _create_tables(bind)
    if "attribution" not in _columns(bind, "conversation_items"):
        op.add_column("conversation_items", sa.Column("attribution", sa.Text(), nullable=True))
    if "attribution" not in _columns(bind, "approval_verdicts"):
        # CompressedText columns are opaque binary in the database.
        op.add_column(
            "approval_verdicts", sa.Column("attribution", sa.LargeBinary(), nullable=True)
        )
    if not _has_index(bind):
        op.create_index(_INDEX, "approval_verdicts", list(_INDEX_COLUMNS))
    if bind.dialect.name == "cockroachdb":
        # CockroachDB publishes schema changes only at commit.
        bind.commit()
        bind.execute(sa.text("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE"))


def downgrade() -> None:
    """Drop both tables, the index and both columns."""
    bind = op.get_bind()
    for table in reversed(_TABLES):
        if sa.inspect(bind).has_table(table):
            op.drop_table(table)
    if _has_index(bind):
        op.drop_index(_INDEX, table_name="approval_verdicts")
    if "attribution" in _columns(bind, "approval_verdicts"):
        with op.batch_alter_table("approval_verdicts") as batch_op:
            batch_op.drop_column("attribution")
    if "attribution" in _columns(bind, "conversation_items"):
        with op.batch_alter_table("conversation_items") as batch_op:
            batch_op.drop_column("attribution")
