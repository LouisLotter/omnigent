"""An invitation that links an existing account, end to end in the browser.

The owner picks an account that has no team membership on Settings → Members
and creates the invitation. The invited person opens the link: the page asks
for that account's password instead of creating a new account. A wrong
password is refused and the invitation stays usable; the right one signs in
and makes the account a member.

Server-side coverage of the same flow lives in
``tests/server/test_team_ingress.py``; this is the browser-level proof.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from playwright.sync_api import Browser, Page, expect

from omnigent.server.accounts_store import SqlAlchemyAccountStore
from omnigent.server.passwords import hash_password
from tests.e2e_ui.auth._accounts_server import (
    ADMIN_PASSWORD,
    ADMIN_USERNAME,
    AccountsServer,
    spawn_accounts_server,
)

_LINKED = "frank"
_LINKED_PASSWORD = "frank-existing-pw-1234"
_LINKED_EMAIL = "frank@example.test"


@pytest.fixture(scope="module")
def server_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("e2e_ui_team_link")


@pytest.fixture(scope="module")
def accounts_server(
    built_spa: None, mock_llm_server_url: str, server_dir: Path
) -> Iterator[AccountsServer]:
    """An accounts-mode server whose seeded admin is the team's owner."""
    yield from spawn_accounts_server(mock_llm_server_url, server_dir)


def _owner_signs_in(page: Page, server: AccountsServer) -> None:
    page.goto(f"{server.public_url}/login")
    page.locator("#login-username").fill(ADMIN_USERNAME)
    page.locator("#login-password").fill(ADMIN_PASSWORD)
    page.get_by_role("button", name="Sign in").click()
    expect(page).not_to_have_url(re.compile(r"/login"), timeout=10_000)


def _team_members(server: AccountsServer) -> dict[str, str]:
    with httpx.Client(base_url=server.base_url, timeout=10.0) as client:
        signed_in = client.post(
            "/auth/login", json={"username": ADMIN_USERNAME, "password": ADMIN_PASSWORD}
        )
        signed_in.raise_for_status()
        team = client.get("/v1/team")
        team.raise_for_status()
        return {entry["user_id"]: entry["state"] for entry in team.json()["members"]}


def test_an_owner_links_an_existing_account_proved_by_its_password(
    accounts_server: AccountsServer, server_dir: Path, page: Page, browser: Browser
) -> None:
    # An account from before the team, with no membership.
    SqlAlchemyAccountStore(f"sqlite:///{server_dir / 'test.db'}").create_user_with_password(
        _LINKED, hash_password(_LINKED_PASSWORD)
    )
    assert _LINKED not in _team_members(accounts_server)

    # The owner picks the account on the Team page and gets the link once.
    _owner_signs_in(page, accounts_server)
    page.goto(f"{accounts_server.public_url}/settings/members")
    page.get_by_role("button", name=re.compile("Invite someone")).click()
    dialog = page.get_by_role("dialog")
    dialog.get_by_label("Email address").fill(_LINKED_EMAIL)
    dialog.get_by_label("Existing account (optional)").select_option(_LINKED)
    dialog.get_by_role("button", name="Create invitation").click()
    invite_url = page.get_by_label("Invitation link").input_value(timeout=10_000)
    assert "/register?invite=" in invite_url

    # The invited person, in a browser without the owner's session.
    context = browser.new_context()
    try:
        invited = context.new_page()
        invited.goto(invite_url)
        expect(invited.get_by_role("heading", name="Sign in to join")).to_be_visible(
            timeout=10_000
        )
        expect(invited.locator("#register-username")).to_have_value(_LINKED)
        expect(invited.locator("#register-confirm")).to_have_count(0)

        invited.locator("#register-password").fill("not-the-password")
        invited.get_by_role("button", name="Sign in and join").click()
        expect(invited.get_by_role("alert")).to_be_visible(timeout=10_000)
        expect(invited).to_have_url(re.compile(r"/register\?invite="))
        assert _LINKED not in _team_members(accounts_server)

        invited.locator("#register-password").fill(_LINKED_PASSWORD)
        invited.get_by_role("button", name="Sign in and join").click()
        expect(invited).not_to_have_url(re.compile(r"/register"), timeout=10_000)
    finally:
        context.close()

    assert _team_members(accounts_server)[_LINKED] == "active"
