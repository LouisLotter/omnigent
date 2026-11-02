"""A runner lists and searches only its person's own threads, in both database layouts.

Share grants never widen a machine principal's listing, whatever the visibility
filter: the store applies the owner restriction itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnigent.db.account_authority import machine_principal_scope
from omnigent.entities.conversation import MessageData, NewConversationItem
from omnigent.server.auth import LEVEL_EDIT, LEVEL_OWNER
from omnigent.stores.conversation_store.sqlalchemy_store import SqlAlchemyConversationStore
from omnigent.stores.permission_store.sqlalchemy_store import SqlAlchemyPermissionStore

_SECRET = "zebracrossing"


def _stores(
    layout: str, db_uri: str, tmp_path: Path
) -> tuple[SqlAlchemyConversationStore, SqlAlchemyPermissionStore]:
    if layout == "single":
        return SqlAlchemyConversationStore(db_uri), SqlAlchemyPermissionStore(db_uri)
    omnigent_uri = f"sqlite:///{tmp_path}/omnigent.db"
    conversations_uri = f"sqlite:///{tmp_path}/conversations.db"
    return (
        SqlAlchemyConversationStore(omnigent_uri, conversations_uri),
        SqlAlchemyPermissionStore(omnigent_uri),
    )


def _threads_mentioning_the_secret(
    store: SqlAlchemyConversationStore, *, archived: bool
) -> tuple[str, str]:
    """One thread naming the secret in its title, one only in its body."""
    by_title = store.create_conversation(title=f"{_SECRET} plan")
    by_body = store.create_conversation(title="notes")
    store.append(
        by_body.id,
        [
            NewConversationItem(
                type="message",
                response_id=f"resp_{by_body.id}",
                data=MessageData(
                    role="user", content=[{"type": "input_text", "text": f"the {_SECRET} notes"}]
                ),
            )
        ],
    )
    if archived:
        for thread in (by_title, by_body):
            store.update_conversation(thread.id, archived=True)
    return by_title.id, by_body.id


@pytest.mark.parametrize("layout", ["single", "split"])
@pytest.mark.parametrize("archived", [False, True], ids=["active", "archived"])
def test_machine_listing_never_includes_a_thread_shared_with_its_person(
    layout: str, archived: bool, db_uri: str, tmp_path: Path
) -> None:
    store, permissions = _stores(layout, db_uri, tmp_path)
    for user in ("alice", "bob"):
        permissions.ensure_user(user)
    own_title, own_body = _threads_mentioning_the_secret(store, archived=archived)
    bobs = _threads_mentioning_the_secret(store, archived=archived)
    for thread in (own_title, own_body):
        permissions.grant("alice", thread, LEVEL_OWNER)
    for thread in bobs:
        permissions.grant("bob", thread, LEVEL_OWNER)
        permissions.grant("alice", thread, LEVEL_EDIT)
    archive = {"include_archived": archived, "archived_only": archived}

    def listed(**filters: object) -> set[str]:
        return {c.id for c in store.list_conversations(limit=100, **archive, **filters).data}

    # A person sees what is shared with them.
    assert listed(accessible_by="alice", shared_only=True) == set(bobs)
    with machine_principal_scope(True):
        expectations: list[tuple[dict[str, object], set[str]]] = [
            ({"accessible_by": "alice", "shared_only": True}, set()),
            ({"accessible_by": "alice", "owned_by": "alice", "shared_only": True}, set()),
            ({"accessible_by": "alice"}, {own_title, own_body}),
            ({"owned_by": "alice"}, {own_title, own_body}),
            (
                {"accessible_by": "alice", "search_query": _SECRET, "search_scope": "all"},
                {own_title, own_body},
            ),
            (
                {"accessible_by": "alice", "search_query": _SECRET, "search_scope": "title"},
                {own_title},
            ),
            (
                {"accessible_by": "alice", "search_query": _SECRET, "search_scope": "content"},
                {own_body},
            ),
        ]
        for filters, expected in expectations:
            assert listed(**filters) == expected, filters
