"""MCP services an administrator manages at run time, beside the registry file's.

A service row holds no secret. Members' logins, an OAuth sign-in in progress and
the OAuth client's own secret live in the credential store's ``connections`` table
under the providers :func:`credential_providers` names, each stamped with the
service's :attr:`McpServiceRecord.credential_version` in its metadata.

Changing a service's destination or login settings replaces that version in the
same transaction that removes the first batch of its logins, so no login saved for
the old settings is ever sent to the new ones. Removal continues in bounded batches;
a crash leaves ``pending_sweep`` set, and :meth:`McpServiceStore.finish_sweeps`
completes it.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from sqlalchemy import and_, delete, or_, select
from sqlalchemy.orm import Session

from omnigent.db import enum_codecs
from omnigent.db.account_authority import require_active_account
from omnigent.db.db_models import (
    SqlConnection,
    SqlMcpRegistryService,
    SqlMcpRegistryServiceTool,
    current_workspace_id,
)
from omnigent.db.utils import (
    get_or_create_engine,
    make_named_managed_session_maker,
    now_epoch,
    run_write_transaction,
)
from omnigent.stores.credential_store.sqlalchemy_store import CredentialStore, SealedSecret

McpServiceAuth = Literal["none", "bearer", "oauth"]
McpClientRegistration = Literal["manual", "dynamic"]
McpTokenEndpointAuth = Literal["none", "client_secret_post", "client_secret_basic"]

#: Connection metadata key holding the service version a login was saved for.
SERVICE_VERSION_KEY = "service_version"
#: Credential-store owner of a service's own secrets (its OAuth client secret).
#: Reserved: no sign-in can produce it (see ``omnigent.server.auth``).
SERVICE_PRINCIPAL = "__mcp_service__"
#: Bounds every write and sweep transaction, and each page of a full load.
BATCH_SIZE = 500
MAX_TOOLS = 200


def member_provider(service_id: str) -> str:
    """Credential-store provider of members' saved logins."""
    return f"mcp:{service_id}"


def pending_provider(service_id: str) -> str:
    """Credential-store provider of an OAuth sign-in in progress."""
    return f"mcp-pending:{service_id}"


def client_provider(service_id: str) -> str:
    """Credential-store provider of the service's OAuth client secret."""
    return f"mcp-client:{service_id}"


def credential_providers(service_id: str) -> tuple[str, str, str]:
    """Every credential-store provider holding a secret for *service_id*."""
    return member_provider(service_id), pending_provider(service_id), client_provider(service_id)


class McpServiceConflict(Exception):
    """The service ID is taken, or the service is still being removed."""


class McpServiceNotFound(Exception):
    """No live run-time service has this ID."""


class McpServiceVersionChanged(Exception):
    """The service changed or went away since the caller read it."""


class McpClientSecretMissing(Exception):
    """A confidential OAuth client would be left without a client secret for its settings."""


@dataclass(frozen=True)
class McpServiceOAuth:
    """A run-time service's OAuth client and endpoints. Holds no secret."""

    authorize_url: str
    token_url: str
    client_id: str
    scopes: tuple[str, ...] = ()
    resource: str | None = None
    registration: McpClientRegistration = "manual"
    token_endpoint_auth: McpTokenEndpointAuth = "none"


@dataclass(frozen=True)
class McpServiceRecord:
    """One run-time MCP service, as stored."""

    service_id: str
    title: str
    url: str
    auth: McpServiceAuth
    timeout: int
    credential_version: str
    tools: tuple[str, ...] = ()
    description: str = ""
    oauth: McpServiceOAuth | None = None
    created_by: str | None = None
    created_at: int = 0
    updated_at: int = 0
    pending_sweep: bool = field(default=False, compare=False)


@dataclass(frozen=True)
class McpServiceSettings:
    """What an administrator sets on a service; everything but its ID and history."""

    title: str
    url: str
    auth: McpServiceAuth
    timeout: int
    tools: tuple[str, ...] = ()
    description: str = ""
    oauth: McpServiceOAuth | None = None

    def same_credentials(self, record: McpServiceRecord) -> bool:
        """Whether logins saved for *record* stay valid under these settings."""
        return (self.url, self.auth, self.oauth) == (record.url, record.auth, record.oauth)


@dataclass(frozen=True)
class McpServiceUpdate:
    """The outcome of changing a service."""

    record: McpServiceRecord
    #: Whether the change replaced the credential version (all logins removed).
    credentials_reset: bool


def _new_version() -> str:
    return uuid.uuid4().hex


def _record(row: SqlMcpRegistryService, tools: Sequence[str]) -> McpServiceRecord:
    oauth = None
    if row.oauth_token_url is not None:
        oauth = McpServiceOAuth(
            authorize_url=row.oauth_authorize_url or "",
            token_url=row.oauth_token_url,
            client_id=row.oauth_client_id or "",
            scopes=tuple((row.oauth_scopes or "").split()),
            resource=row.oauth_resource,
            registration=cast(
                McpClientRegistration,
                enum_codecs.decode_mcp_oauth_client_registration(
                    row.oauth_client_registration or 1
                ),
            ),
            token_endpoint_auth=cast(
                McpTokenEndpointAuth,
                enum_codecs.decode_mcp_oauth_token_endpoint_auth(
                    row.oauth_token_endpoint_auth or 1
                ),
            ),
        )
    return McpServiceRecord(
        service_id=row.service_id,
        title=row.title,
        description=row.description or "",
        url=row.url,
        auth=cast(McpServiceAuth, enum_codecs.decode_mcp_service_auth(row.auth)),
        timeout=row.timeout_seconds,
        credential_version=row.credential_version,
        oauth=oauth,
        tools=tuple(sorted(tools)),
        created_by=row.created_by,
        created_at=row.created_at,
        updated_at=row.updated_at,
        pending_sweep=row.pending_sweep,
    )


def _apply_settings(row: SqlMcpRegistryService, settings: McpServiceSettings) -> None:
    row.title = settings.title
    row.description = settings.description or None
    row.url = settings.url
    row.auth = enum_codecs.encode_mcp_service_auth(settings.auth)
    row.timeout_seconds = settings.timeout
    oauth = settings.oauth
    row.oauth_authorize_url = oauth.authorize_url if oauth else None
    row.oauth_token_url = oauth.token_url if oauth else None
    row.oauth_client_id = oauth.client_id if oauth else None
    row.oauth_scopes = " ".join(oauth.scopes) if oauth and oauth.scopes else None
    row.oauth_resource = oauth.resource if oauth else None
    row.oauth_client_registration = (
        enum_codecs.encode_mcp_oauth_client_registration(oauth.registration) if oauth else None
    )
    row.oauth_token_endpoint_auth = (
        enum_codecs.encode_mcp_oauth_token_endpoint_auth(oauth.token_endpoint_auth)
        if oauth
        else None
    )


def _metadata(metadata_json: str | None) -> dict[str, Any]:
    try:
        metadata = json.loads(metadata_json or "")
    except ValueError:
        return {}
    return metadata if isinstance(metadata, dict) else {}


def _stamp(metadata_json: str | None) -> str | None:
    value = _metadata(metadata_json).get(SERVICE_VERSION_KEY)
    return value if isinstance(value, str) else None


def _lock_accounts(session: Session) -> None:
    """Lock the acting account (and the service principal) before the service and its
    connections: account deletion locks accounts first, then deletes connections."""
    require_active_account(session, SERVICE_PRINCIPAL)


def _confidential(settings: McpServiceSettings) -> bool:
    """Whether the settings' OAuth client authenticates with a client secret."""
    return settings.oauth is not None and settings.oauth.token_endpoint_auth != "none"


class McpServiceStore:
    """SQLAlchemy persistence for run-time MCP services and their login cleanup.

    :param storage_location: SQLAlchemy database URI (shares the pool with the
        other stores via :func:`get_or_create_engine`).
    """

    def __init__(self, storage_location: str) -> None:
        self.storage_location = storage_location
        self._engine = get_or_create_engine(storage_location)
        self._session = make_named_managed_session_maker(
            self._engine, query_name_prefix="omnigent.mcp_service_store"
        )
        self._session_immediate = make_named_managed_session_maker(
            self._engine, query_name_prefix="omnigent.mcp_service_store", immediate=True
        )

    # ── Reads ───────────────────────────────────────────

    def _tools_for(self, session: Session, service_ids: Sequence[str]) -> dict[str, list[str]]:
        tools: dict[str, list[str]] = {service_id: [] for service_id in service_ids}
        if not service_ids:
            return tools
        rows = session.execute(
            select(SqlMcpRegistryServiceTool.service_id, SqlMcpRegistryServiceTool.tool_name)
            .where(
                SqlMcpRegistryServiceTool.workspace_id == current_workspace_id(),
                SqlMcpRegistryServiceTool.service_id.in_(service_ids),
            )
            .limit(len(service_ids) * MAX_TOOLS)
        ).all()
        for service_id, tool_name in rows:
            tools[service_id].append(tool_name)
        return tools

    def list_page(self, *, after: str = "", limit: int = 100) -> list[McpServiceRecord]:
        """Live services ordered by ID, after the *after* cursor."""
        limit = max(1, min(limit, 100))
        with self._session("list_services") as session:
            rows = (
                session.execute(
                    select(SqlMcpRegistryService)
                    .where(
                        SqlMcpRegistryService.workspace_id == current_workspace_id(),
                        SqlMcpRegistryService.service_id > after,
                    )
                    .order_by(SqlMcpRegistryService.service_id)
                    .limit(limit)
                )
                .scalars()
                .all()
            )
            live = [row for row in rows if row.deleted_at is None]
            tools = self._tools_for(session, [row.service_id for row in live])
            return [_record(row, tools[row.service_id]) for row in live]

    def scan(self) -> tuple[list[McpServiceRecord], list[str]]:
        """Every live service, and the IDs with logins left to remove, in bounded pages."""
        services: list[McpServiceRecord] = []
        pending: list[str] = []
        after = ""
        while True:
            with self._session("scan_services") as session:
                rows = (
                    session.execute(
                        select(SqlMcpRegistryService)
                        .where(
                            SqlMcpRegistryService.workspace_id == current_workspace_id(),
                            SqlMcpRegistryService.service_id > after,
                        )
                        .order_by(SqlMcpRegistryService.service_id)
                        .limit(100)
                    )
                    .scalars()
                    .all()
                )
                live = [row for row in rows if row.deleted_at is None]
                tools = self._tools_for(session, [row.service_id for row in live])
                services.extend(_record(row, tools[row.service_id]) for row in live)
                pending.extend(row.service_id for row in rows if row.pending_sweep)
            if len(rows) < 100:
                return services, pending
            after = rows[-1].service_id

    def get(self, service_id: str) -> McpServiceRecord | None:
        """One live service, or ``None``."""
        with self._session("get_service") as session:
            row = session.get(SqlMcpRegistryService, (current_workspace_id(), service_id))
            if row is None or row.deleted_at is not None:
                return None
            return _record(row, self._tools_for(session, [service_id])[service_id])

    # ── Writes ──────────────────────────────────────────

    @staticmethod
    def _lock(session: Session, service_id: str) -> SqlMcpRegistryService | None:
        return session.execute(
            select(SqlMcpRegistryService)
            .where(
                SqlMcpRegistryService.workspace_id == current_workspace_id(),
                SqlMcpRegistryService.service_id == service_id,
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        ).scalar_one_or_none()

    @staticmethod
    def _replace_tools(session: Session, service_id: str, tools: Sequence[str]) -> None:
        session.execute(
            delete(SqlMcpRegistryServiceTool).where(
                SqlMcpRegistryServiceTool.workspace_id == current_workspace_id(),
                SqlMcpRegistryServiceTool.service_id == service_id,
            )
        )
        for tool in sorted(set(tools)):
            session.add(SqlMcpRegistryServiceTool(service_id=service_id, tool_name=tool))

    @staticmethod
    def _check_sealed(service_id: str, sealed: SealedSecret | None) -> None:
        if sealed is not None and (sealed.user_id, sealed.provider, sealed.account_id) != (
            SERVICE_PRINCIPAL,
            client_provider(service_id),
            "",
        ):
            raise ValueError("the sealed secret is not this service's client secret")

    @staticmethod
    def _client_row(session: Session, service_id: str) -> SqlConnection | None:
        return session.execute(
            select(SqlConnection)
            .where(
                SqlConnection.workspace_id == current_workspace_id(),
                SqlConnection.user_id == SERVICE_PRINCIPAL,
                SqlConnection.provider == client_provider(service_id),
                SqlConnection.account_id == "",
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        ).scalar_one_or_none()

    def create(
        self,
        service_id: str,
        settings: McpServiceSettings,
        *,
        created_by: str | None,
        client_secret: SealedSecret | None = None,
    ) -> McpServiceRecord:
        """Add a service. :class:`McpServiceConflict` if the ID is taken.

        :param client_secret: The OAuth client's secret, sealed with
            :meth:`CredentialStore.seal`; saved in the same transaction.
        """
        if len(settings.tools) > MAX_TOOLS:
            raise ValueError(f"A service allows at most {MAX_TOOLS} tools")
        self._check_sealed(service_id, client_secret)
        if _confidential(settings) and client_secret is None:
            raise McpClientSecretMissing(service_id)
        now = now_epoch()

        def write(session: Session) -> McpServiceRecord:
            _lock_accounts(session)
            if self._lock(session, service_id) is not None:
                raise McpServiceConflict(service_id)
            row = SqlMcpRegistryService(
                service_id=service_id,
                credential_version=_new_version(),
                pending_sweep=False,
                created_by=created_by,
                created_at=now,
                updated_at=now,
            )
            _apply_settings(row, settings)
            session.add(row)
            self._replace_tools(session, service_id, settings.tools)
            session.flush()
            # Logins left by an earlier service with this ID carry its version: never valid.
            more, _ = self._delete_logins(session, service_id, keep_version=row.credential_version)
            row.pending_sweep = more
            if client_secret is not None:
                CredentialStore.write_sealed(
                    session, client_secret, stamped({}, row.credential_version)
                )
            return _record(row, settings.tools)

        record = run_write_transaction(self._session_immediate, "create_service", write)
        if record.pending_sweep:
            self.finish_sweep(service_id)
        return record

    def update(
        self,
        service_id: str,
        settings: McpServiceSettings,
        *,
        expected_version: str | None = None,
        client_secret: SealedSecret | None = None,
        keep_client_secret: bool = False,
    ) -> McpServiceUpdate:
        """Replace a service's settings.

        A change to its destination or login settings replaces the credential
        version and removes its logins, starting in the same transaction.

        :param expected_version: Refuse (:class:`McpServiceVersionChanged`) unless the
            stored version still matches, e.g. after an OAuth client was registered
            for the settings the caller read.
        :param client_secret: A new OAuth client secret (from :meth:`CredentialStore.seal`),
            saved in the same transaction for the version this write leaves.
        :param keep_client_secret: Keep the saved client secret, which must belong to
            the version the caller read; the caller keeps it only while the token URL
            and client ID stay the same.
        :raises McpClientSecretMissing: A confidential client would have no secret.
        """
        if len(settings.tools) > MAX_TOOLS:
            raise ValueError(f"A service allows at most {MAX_TOOLS} tools")
        self._check_sealed(service_id, client_secret)
        now = now_epoch()

        def write(session: Session) -> McpServiceUpdate:
            _lock_accounts(session)
            row = self._lock(session, service_id)
            if row is None or row.deleted_at is not None:
                raise McpServiceNotFound(service_id)
            if expected_version is not None and row.credential_version != expected_version:
                raise McpServiceVersionChanged(service_id)
            saved_client = self._client_row(session, service_id)
            saved_valid = (
                saved_client is not None
                and _stamp(saved_client.metadata_json) == row.credential_version
            )
            keep = keep_client_secret and client_secret is None
            if keep and not saved_valid:
                raise McpClientSecretMissing(service_id)
            reset = not settings.same_credentials(_record(row, ()))
            if (
                _confidential(settings)
                and client_secret is None
                and not (keep or (saved_valid and not reset))
            ):
                raise McpClientSecretMissing(service_id)
            _apply_settings(row, settings)
            row.updated_at = now
            self._replace_tools(session, service_id, settings.tools)
            if reset:
                row.credential_version = _new_version()
                more, _ = self._delete_logins(
                    session, service_id, keep_version=None, spare_client=keep
                )
                row.pending_sweep = row.pending_sweep or more
                if keep:
                    assert saved_client is not None
                    saved_client.metadata_json = json.dumps(
                        stamped(_metadata(saved_client.metadata_json), row.credential_version)
                    )
            if client_secret is not None:
                CredentialStore.write_sealed(
                    session, client_secret, stamped({}, row.credential_version)
                )
            session.flush()
            return McpServiceUpdate(_record(row, settings.tools), credentials_reset=reset)

        result = run_write_transaction(self._session_immediate, "update_service", write)
        if result.record.pending_sweep:
            self.finish_sweep(service_id)
        return result

    def delete(self, service_id: str) -> bool:
        """Remove a service and its logins. Returns whether a live service went."""
        now = now_epoch()

        def write(session: Session) -> tuple[bool, bool]:
            _lock_accounts(session)
            row = self._lock(session, service_id)
            if row is None or row.deleted_at is not None:
                return False, False
            self._replace_tools(session, service_id, ())
            remaining, _ = self._delete_logins(session, service_id, keep_version=None)
            if remaining:
                row.deleted_at = now
                row.pending_sweep = True
                row.credential_version = _new_version()
            else:
                session.delete(row)
            return True, remaining

        removed, remaining = run_write_transaction(
            self._session_immediate, "delete_service", write
        )
        if remaining:
            self.finish_sweep(service_id)
        return removed

    def _delete_logins(
        self,
        session: Session,
        service_id: str,
        *,
        keep_version: str | None,
        after: tuple[str, str, str] | None = None,
        spare_client: bool = False,
    ) -> tuple[bool, tuple[str, str, str] | None]:
        """Delete up to one batch of the service's logins not saved for *keep_version*.

        With *spare_client*, the OAuth client's secret is left for the caller to restamp.
        Returns whether rows may remain past this batch, and the batch's last key.
        """
        workspace_id = current_workspace_id()
        providers = [
            provider
            for provider in credential_providers(service_id)
            if not (spare_client and provider == client_provider(service_id))
        ]
        query = (
            select(
                SqlConnection.provider,
                SqlConnection.user_id,
                SqlConnection.account_id,
                SqlConnection.metadata_json,
            )
            .where(
                SqlConnection.workspace_id == workspace_id,
                SqlConnection.provider.in_(providers),
            )
            .order_by(SqlConnection.provider, SqlConnection.user_id, SqlConnection.account_id)
            .limit(BATCH_SIZE)
        )
        if after is not None:
            provider, user_id, account_id = after
            # Expanded rather than a tuple comparison, which MySQL cannot seek on.
            query = query.where(
                or_(
                    SqlConnection.provider > provider,
                    and_(
                        SqlConnection.provider == provider,
                        or_(
                            SqlConnection.user_id > user_id,
                            and_(
                                SqlConnection.user_id == user_id,
                                SqlConnection.account_id > account_id,
                            ),
                        ),
                    ),
                )
            )
        rows = session.execute(query).all()
        for provider, user_id, account_id, metadata_json in rows:
            if keep_version is not None and _stamp(metadata_json) == keep_version:
                continue
            session.execute(
                delete(SqlConnection).where(
                    SqlConnection.workspace_id == workspace_id,
                    SqlConnection.user_id == user_id,
                    SqlConnection.provider == provider,
                    SqlConnection.account_id == account_id,
                )
            )
        last = (rows[-1][0], rows[-1][1], rows[-1][2]) if rows else None
        return len(rows) == BATCH_SIZE, last

    def finish_sweep(self, service_id: str) -> None:
        """Remove what an interrupted change or removal left, one bounded batch per write."""
        cursor: tuple[str, str, str] | None = None
        while True:

            def write(
                session: Session, after: tuple[str, str, str] | None = cursor
            ) -> tuple[bool, tuple[str, str, str] | None]:
                row = self._lock(session, service_id)
                if row is None:
                    return False, None
                tombstone = row.deleted_at is not None
                more, last = self._delete_logins(
                    session,
                    service_id,
                    keep_version=None if tombstone else row.credential_version,
                    after=after,
                )
                if more:
                    return True, last
                if tombstone:
                    session.delete(row)
                else:
                    row.pending_sweep = False
                return False, None

            more, cursor = run_write_transaction(self._session_immediate, "sweep_logins", write)
            if not more:
                return

    def finish_sweeps(self, service_ids: Sequence[str]) -> None:
        """:meth:`finish_sweep` for each of *service_ids*."""
        for service_id in service_ids:
            self.finish_sweep(service_id)

    # ── Credential-write precondition ───────────────────

    @staticmethod
    def require_version(session: Session, service_id: str, version: str) -> None:
        """Inside a login write: lock the service and require *version*.

        Serializes the write with changes to the service, so a login saved for
        settings an administrator just replaced is refused rather than left behind.
        """
        row = McpServiceStore._lock(session, service_id)
        if row is None or row.deleted_at is not None or row.credential_version != version:
            raise McpServiceVersionChanged(service_id)


def version_guard(service_id: str, version: str) -> Callable[[Session], None]:
    """A credential-store write precondition for :meth:`McpServiceStore.require_version`."""

    def guard(session: Session) -> None:
        McpServiceStore.require_version(session, service_id, version)

    return guard


def stamped(metadata: Mapping[str, Any], version: str) -> dict[str, Any]:
    """*metadata* with the service version a login was saved for."""
    return {**metadata, SERVICE_VERSION_KEY: version}
