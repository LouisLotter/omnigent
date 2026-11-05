/**
 * One member's page, for the team's owners and admins (``/settings/members/<member id>``):
 * their computers, connections, threads and account.
 *
 * An owner or admin acts on the member's things, never as the member. The
 * computers stay theirs, and so do their sign-ins: there is no sign-in or
 * connect here, only whether they are signed in or connected, and Disconnect.
 * The server enforces all of it; this page only offers what it allows.
 */

import { useCallback, useEffect, useState } from "react";
import { ArrowLeftIcon, ExternalLinkIcon, Loader2Icon, PlusIcon } from "lucide-react";

import { AddComputerDialog } from "@/components/computers/AddComputerDialog";
import { DeleteComputerDialog } from "@/components/computers/DeleteComputerDialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { Host } from "@/hooks/useHosts";
import { useMemberHosts, useStartPersistentHost } from "@/hooks/usePersistentHosts";
import { withBasePath } from "@/lib/basePath";
import { sameOriginPage } from "@/lib/capabilities";
import {
  describePersistentHostState,
  isPersistentHost,
  isPersistentHostChanging,
  isPersistentHostFailed,
  persistentHostManageUrl,
} from "@/lib/persistentHosts";
import {
  type MemberConnections,
  type RemovedThread,
  type TeamMemberEntry,
  type TeamState,
  disconnectMemberProvider,
  disconnectMemberService,
  getMemberConnections,
  getMemberThreads,
} from "@/lib/teamApi";
import { MemberAccountActions } from "@/pages/settings/MemberAccountActions";
import { ROLE_LABELS } from "@/pages/settings/TeamPanel";

const PROVIDER_LABELS: Record<string, string> = { github: "GitHub", databricks: "Databricks" };

interface MemberPageProps {
  memberId: string;
  state: TeamState;
  /** Providers that create computers, from `/v1/info`. */
  providers: string[];
  /** The computers' overview page on this origin, or null. */
  computersPage: string | null;
  onChanged: () => Promise<void>;
  onResetPassword: (userId: string) => void;
  onBack: () => void;
}

export function MemberPage({
  memberId,
  state,
  providers,
  computersPage,
  onChanged,
  onResetPassword,
  onBack,
}: MemberPageProps) {
  const member = state.members.find(
    (entry) => entry.member_id === memberId && entry.state !== "removed",
  );
  return (
    <section aria-labelledby="member-heading" data-member-page={memberId}>
      <Button variant="ghost" size="sm" className="mb-4 gap-1.5" onClick={onBack}>
        <ArrowLeftIcon className="size-4" /> Members
      </Button>
      {member === undefined ? (
        <p className="text-ui text-muted-foreground" data-testid="member-missing">
          This person is not on the team.
        </p>
      ) : (
        <>
          <div className="flex flex-wrap items-center gap-2">
            <h2 id="member-heading" className="text-lg font-semibold">
              {member.user_id}
            </h2>
            <Badge variant={member.role === "member" ? "secondary" : "default"}>
              {ROLE_LABELS[member.role]}
            </Badge>
            {member.state === "suspended" ? (
              <Badge variant="destructive">Suspended</Badge>
            ) : (
              <Badge variant="outline">Active</Badge>
            )}
          </div>
          <p className="mt-1 text-ui text-muted-foreground">
            You manage {member.user_id}&apos;s computers and connections here. They stay{" "}
            {member.user_id}&apos;s, and {member.user_id} signs in to their accounts themselves.
          </p>
          <ComputersSection member={member} providers={providers} computersPage={computersPage} />
          <ConnectionsSection member={member} />
          <ThreadsSection member={member} />
          <section className="mt-8" aria-labelledby="member-account-heading">
            <h3 id="member-account-heading" className="mb-2 text-ui font-semibold">
              Account
            </h3>
            <div className="flex flex-wrap items-center gap-1" data-testid="member-account">
              <MemberAccountActions
                me={state.me}
                member={member}
                onChanged={onChanged}
                onResetPassword={onResetPassword}
                onRemoved={onBack}
              />
            </div>
          </section>
        </>
      )}
    </section>
  );
}

function ComputersSection({
  member,
  providers,
  computersPage,
}: {
  member: TeamMemberEntry;
  providers: string[];
  computersPage: string | null;
}) {
  const hosts = useMemberHosts(member.member_id);
  const [addOpen, setAddOpen] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<Host | null>(null);
  const computers = (hosts.data ?? []).filter(isPersistentHost);
  const suspended = member.state === "suspended";
  return (
    <section className="mt-6" aria-labelledby="member-computers-heading">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <h3 id="member-computers-heading" className="text-ui font-semibold">
          Computers
        </h3>
        {providers.length > 0 && !suspended && (
          <Button
            size="sm"
            className="gap-1.5"
            onClick={() => setAddOpen(true)}
            data-testid="member-add-computer"
          >
            <PlusIcon className="size-4" /> Add a computer for {member.user_id}
          </Button>
        )}
      </div>
      {suspended && (
        <p className="mb-2 text-ui text-muted-foreground" data-testid="member-suspended-note">
          {member.user_id} is suspended, so their computers can&apos;t connect. You can delete them;
          reinstate {member.user_id} to use them again.
        </p>
      )}
      {hosts.isError && !hosts.data ? (
        <p role="alert" className="text-ui">
          Couldn&apos;t load {member.user_id}&apos;s computers.{" "}
          <Button variant="link" onClick={() => void hosts.refetch()}>
            Retry
          </Button>
        </p>
      ) : hosts.data === undefined ? (
        <p className="text-ui text-muted-foreground">Loading computers…</p>
      ) : computers.length === 0 ? (
        <p className="text-ui text-muted-foreground" data-testid="member-computers-empty">
          No computers yet.
        </p>
      ) : (
        <ul className="divide-y divide-border rounded-md border border-border">
          {computers.map((host) => (
            <MemberComputerRow
              key={host.host_id}
              host={host}
              wakeable={!suspended}
              computersPage={computersPage}
              onDelete={() => setDeleteTarget(host)}
            />
          ))}
        </ul>
      )}
      <AddComputerDialog
        open={addOpen}
        onOpenChange={setAddOpen}
        providers={providers}
        ownerUserId={member.user_id}
      />
      <DeleteComputerDialog
        host={deleteTarget}
        onClose={() => setDeleteTarget(null)}
        ownerName={member.user_id}
      />
    </section>
  );
}

/** The computer's provider page: its own when Omnigent knows it, else the overview. */
function managePage(host: Host, computersPage: string | null): string | null {
  const own = persistentHostManageUrl(host);
  const page = own !== null ? sameOriginPage(own, window.location.origin) : null;
  return page ?? computersPage;
}

function MemberComputerRow({
  host,
  wakeable,
  computersPage,
  onDelete,
}: {
  host: Host;
  wakeable: boolean;
  computersPage: string | null;
  onDelete: () => void;
}) {
  const start = useStartPersistentHost();
  const state = describePersistentHostState(host);
  const status = host.persistent_status?.state;
  const page = managePage(host, computersPage);
  const failed = isPersistentHostFailed(host);
  return (
    <li
      className="flex flex-wrap items-center justify-between gap-x-6 gap-y-2 p-3"
      data-testid={`member-computer-${host.host_id}`}
    >
      <div className="flex min-w-0 flex-1 flex-col gap-1">
        <span className="truncate text-ui font-medium">{host.name}</span>
        {state && (
          <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
            {isPersistentHostChanging(host) && (
              <Loader2Icon aria-hidden className="size-3 shrink-0 animate-spin" />
            )}
            <span className={failed ? "text-destructive" : undefined}>{state.label}</span>
            {state.detail && <span>· {state.detail}</span>}
          </span>
        )}
        {start.error && (
          <span role="alert" className="text-xs text-destructive">
            {start.error.message}
          </span>
        )}
      </div>
      <div className="flex shrink-0 flex-wrap items-center gap-2">
        {page !== null && (
          <a
            href={page}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 px-2 text-ui underline underline-offset-2 hover:no-underline"
            data-testid="member-computer-manage"
          >
            Manage
            <ExternalLinkIcon aria-hidden className="size-3 shrink-0" />
          </a>
        )}
        {wakeable && (status === "asleep" || failed) && (
          <Button
            variant="outline"
            size="sm"
            loading={start.isPending}
            onClick={() => start.mutate(host.host_id)}
          >
            {failed ? "Start again" : "Wake"}
          </Button>
        )}
        <Button variant="ghost" size="sm" onClick={onDelete}>
          Delete
        </Button>
      </div>
    </li>
  );
}

function connectionState(connected: boolean, needsReconnect = false): string {
  if (needsReconnect) return "Needs reconnecting";
  return connected ? "Connected" : "Not connected";
}

function ConnectionsSection({ member }: { member: TeamMemberEntry }) {
  const [connections, setConnections] = useState<MemberConnections | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState<string | null>(null);

  const load = useCallback(async () => {
    const result = await getMemberConnections(member.member_id);
    if (!result.ok) {
      setError(result.error);
      return;
    }
    setError(null);
    setConnections({ services: result.services, providers: result.providers });
  }, [member.member_id]);

  useEffect(() => {
    void load();
  }, [load]);

  async function disconnect(key: string, action: () => Promise<{ ok: boolean; error?: string }>) {
    setPending(key);
    const result = await action();
    setPending(null);
    if (!result.ok) {
      setError(result.error ?? "The connection was not removed.");
      return;
    }
    await load();
  }

  const rows = connections
    ? [
        ...connections.providers.map((provider) => ({
          key: `provider:${provider.provider}`,
          label: PROVIDER_LABELS[provider.provider] ?? provider.provider,
          state: connectionState(provider.connected),
          removable: provider.connected,
          remove: () => disconnectMemberProvider(member.member_id, provider.provider),
        })),
        ...connections.services.map((service) => ({
          key: `mcp:${service.id}`,
          label: `${service.title} (MCP)`,
          state: connectionState(service.connected, service.needs_reconnect),
          removable: service.disconnectable,
          remove: () => disconnectMemberService(member.member_id, service.id),
        })),
      ]
    : [];

  return (
    <section className="mt-8" aria-labelledby="member-connections-heading">
      <h3 id="member-connections-heading" className="mb-2 text-ui font-semibold">
        Connections
      </h3>
      {error !== null && (
        <p role="alert" className="mb-2 text-ui text-destructive">
          {error}
        </p>
      )}
      {connections === null ? (
        error === null && <p className="text-ui text-muted-foreground">Loading connections…</p>
      ) : rows.length === 0 ? (
        <p className="text-ui text-muted-foreground">No services to connect.</p>
      ) : (
        <ul className="divide-y divide-border rounded-md border border-border">
          {rows.map((row) => (
            <li
              key={row.key}
              className="flex items-center justify-between gap-4 px-3 py-2 text-ui"
              data-connection={row.key}
            >
              <span>
                <span className="font-medium">{row.label}</span>{" "}
                <span className="text-muted-foreground">· {row.state}</span>
              </span>
              {row.removable && (
                <Button
                  variant="ghost"
                  size="xs"
                  loading={pending === row.key}
                  disabled={pending !== null}
                  onClick={() => void disconnect(row.key, row.remove)}
                >
                  Disconnect
                </Button>
              )}
            </li>
          ))}
        </ul>
      )}
      <p className="mt-2 text-sm text-muted-foreground" data-testid="member-connections-note">
        {member.user_id} connects these themselves, from their own Settings → MCP and Settings →
        Sandbox Integrations.
      </p>
    </section>
  );
}

function ThreadsSection({ member }: { member: TeamMemberEntry }) {
  const [threads, setThreads] = useState<RemovedThread[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let current = true;
    void getMemberThreads(member.member_id).then((result) => {
      if (!current) return;
      if (result.ok) setThreads(result.threads);
      else setError(result.error);
    });
    return () => {
      current = false;
    };
  }, [member.member_id]);

  return (
    <section className="mt-8" aria-labelledby="member-threads-heading">
      <h3 id="member-threads-heading" className="mb-2 text-ui font-semibold">
        Threads
      </h3>
      {error !== null ? (
        <p role="alert" className="text-ui text-destructive">
          {error}
        </p>
      ) : threads === null ? (
        <p className="text-ui text-muted-foreground">Loading threads…</p>
      ) : threads.length === 0 ? (
        <p className="text-ui text-muted-foreground">No threads.</p>
      ) : (
        <ul className="list-disc pl-6 text-ui" data-testid="member-threads">
          {threads.map((thread) => (
            <li key={thread.session_id}>
              <a className="underline" href={withBasePath(`/c/${thread.session_id}`)}>
                {thread.title ?? "Untitled thread"}
              </a>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
