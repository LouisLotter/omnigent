/**
 * The team section of Settings → Members: members with their roles and
 * states, pending invitations, removed members, and the owner's and admins'
 * actions.
 *
 * Who may do what mirrors the server, which enforces it: an owner invites
 * admins and members, changes roles and suspends, reinstates or removes
 * anyone but themselves and the installation's own owner; an admin invites
 * members and suspends, reinstates or removes members. Hiding a button is
 * only convenience.
 */

import { useState } from "react";
import {
  KeyRoundIcon,
  PauseIcon,
  PlayIcon,
  RefreshCwIcon,
  Trash2Icon,
  UserPlusIcon,
  XIcon,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import {
  type RemovedThread,
  type TeamMemberEntry,
  type TeamRemovalEntry,
  type TeamRole,
  type TeamState,
  changeMemberRole,
  getRemoval,
  inviteToTeam,
  removeMember,
  revokeInvitation,
  setMemberSuspended,
} from "@/lib/teamApi";
import { withBasePath } from "@/lib/basePath";

/** The default account list: one stable reference, so renders compare equal. */
const NO_ACCOUNTS: readonly string[] = [];

interface TeamPanelProps {
  state: TeamState;
  /** Every account on the server; an owner may link one without a membership. */
  accounts?: readonly string[];
  /** Reload the team after a change. */
  onChanged: () => Promise<void>;
  onResetPassword: (userId: string) => void;
  /** Renders an invite link for the current origin and base path. */
  rebaseUrl: (url: string) => string;
}

const ROLE_LABELS: Record<TeamRole, string> = { owner: "Owner", admin: "Admin", member: "Member" };

function mayChangeState(me: TeamMemberEntry, target: TeamMemberEntry): boolean {
  if (target.member_id === me.member_id) return false;
  return me.role === "owner" || target.role === "member";
}

function mayRemove(me: TeamMemberEntry, target: TeamMemberEntry): boolean {
  return mayChangeState(me, target) && target.installation_owner !== true;
}

function hasPendingStep(removal: TeamRemovalEntry): boolean {
  const { steps } = removal;
  return (
    steps.member_group === "pending" ||
    steps.computers.state === "pending" ||
    steps.devices.state === "pending"
  );
}

/** One removal's steps, each saying what is done and what isn't. */
function RemovalSteps({ removal }: { removal: TeamRemovalEntry }) {
  const { steps } = removal;
  const computers = steps.computers;
  return (
    <ul className="mt-1 space-y-0.5 text-sm text-muted-foreground" data-removal-steps>
      <li data-step="access">
        ✓ Access blocked: every sign-in, CLI login, laptop, runner and invitation of theirs stopped
        working.
      </li>
      {steps.member_group !== "not_used" && (
        <li data-step="member_group" data-state={steps.member_group}>
          {steps.member_group === "done"
            ? "✓ The installation's sign-in no longer lets them in."
            : "… Telling the installation's sign-in to stop letting them in. This is retried until it is confirmed."}
        </li>
      )}
      <li data-step="computers" data-state={computers.state}>
        {computers.total === 0 ? (
          "✓ No cloud computers."
        ) : (
          <>
            {computers.state === "done" ? "✓" : "…"} Cloud computers handed over to stop and be
            deleted: {computers.handed_over} of {computers.total}.
            {computers.state === "pending" && " The rest are retried."}{" "}
            {computers.fleet_url !== null ? (
              <a className="underline" href={computers.fleet_url} data-link="fleet">
                Delete them on the computers page
              </a>
            ) : (
              "Delete them where your computers are managed."
            )}
          </>
        )}
      </li>
      <li data-step="devices" data-state={steps.devices.state}>
        {steps.devices.state === "not_used"
          ? "Laptop device registrations: none to revoke; laptops are not registered per member yet."
          : steps.devices.state === "done"
            ? "✓ Laptop device registrations revoked."
            : `… Laptop device registrations still to revoke: ${steps.devices.remaining}.`}
      </li>
      <li data-step="threads">
        {removal.thread_count === 1 ? "1 thread" : `${removal.thread_count} threads`} kept, readable
        by owners and admins. Nothing was moved to anyone else.
      </li>
    </ul>
  );
}

export function TeamPanel({
  state,
  accounts = NO_ACCOUNTS,
  onChanged,
  onResetPassword,
  rebaseUrl,
}: TeamPanelProps) {
  const { me } = state;
  const isOwner = me.role === "owner";
  const [showInvite, setShowInvite] = useState(false);
  const [email, setEmail] = useState("");
  const [inviteRole, setInviteRole] = useState<"admin" | "member">("member");
  // An existing account to link instead of creating one (owners only).
  const [linkAccount, setLinkAccount] = useState("");
  const [inviteLink, setInviteLink] = useState<{ url: string; email: string } | null>(null);
  const [stateChange, setStateChange] = useState<{
    member: TeamMemberEntry;
    suspend: boolean;
  } | null>(null);
  const [removal, setRemoval] = useState<TeamMemberEntry | null>(null);
  // The threads of the removal whose list is open, by its member id.
  const [threads, setThreads] = useState<{ memberId: string; list: RemovedThread[] } | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function showThreads(memberId: string) {
    if (threads?.memberId === memberId) {
      setThreads(null);
      return;
    }
    const result = await getRemoval(memberId);
    if (!result.ok) {
      setError(result.error);
      return;
    }
    setThreads({ memberId, list: result.threads });
  }

  async function run(action: () => Promise<{ ok: boolean; error?: string }>): Promise<boolean> {
    setPending(true);
    setError(null);
    const result = await action();
    setPending(false);
    if (!result.ok) {
      setError(result.error ?? "The change failed.");
      return false;
    }
    await onChanged();
    return true;
  }

  async function onInvite() {
    setPending(true);
    setError(null);
    const result = linkAccount
      ? await inviteToTeam(email, inviteRole, linkAccount)
      : await inviteToTeam(email, inviteRole);
    setPending(false);
    if (!result.ok) {
      setError(result.error);
      return;
    }
    setShowInvite(false);
    setInviteLink({ url: rebaseUrl(result.invite_url), email: result.invitation.email });
    setEmail("");
    setInviteRole("member");
    setLinkAccount("");
    await onChanged();
  }

  const activeMembers = state.members.filter((m) => m.state !== "removed");
  const memberAccounts = new Set(activeMembers.map((m) => m.user_id));
  const linkable = accounts.filter((id) => !memberAccounts.has(id));

  return (
    <section aria-labelledby="team-heading">
      <div className="mb-4 flex items-center justify-between">
        <h2 id="team-heading" className="text-lg font-semibold">
          Team
        </h2>
        <Button onClick={() => setShowInvite(true)}>
          <UserPlusIcon /> Invite someone
        </Button>
      </div>

      {error !== null && !showInvite && stateChange === null && removal === null && (
        <div
          role="alert"
          className="mb-4 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-ui text-destructive"
        >
          {error}
        </div>
      )}

      <div className="overflow-hidden rounded-md border border-border">
        <table className="w-full text-ui">
          <thead className="bg-muted/40 text-left text-sm uppercase text-muted-foreground">
            <tr>
              <th className="px-3 py-2 font-medium">Username</th>
              <th className="px-3 py-2 font-medium">Role</th>
              <th className="px-3 py-2 font-medium">State</th>
              <th className="px-3 py-2 text-right font-medium">Actions</th>
            </tr>
          </thead>
          <tbody>
            {activeMembers.map((member) => (
              <tr
                key={member.member_id}
                className="border-t border-border"
                data-member={member.user_id}
              >
                <td className="px-3 py-2 align-middle">
                  <span className="font-medium">{member.user_id}</span>
                  {member.member_id === me.member_id && (
                    <span className="ml-2 text-sm text-muted-foreground">(you)</span>
                  )}
                </td>
                <td className="px-3 py-2 align-middle">
                  {isOwner ? (
                    <select
                      aria-label={`Role of ${member.user_id}`}
                      className="rounded-md border border-border bg-background px-2 py-1 text-ui"
                      value={member.role}
                      disabled={pending}
                      onChange={(e) =>
                        void run(() =>
                          changeMemberRole(member.member_id, e.target.value as TeamRole),
                        )
                      }
                    >
                      {(Object.keys(ROLE_LABELS) as TeamRole[]).map((role) => (
                        <option key={role} value={role}>
                          {ROLE_LABELS[role]}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <Badge variant={member.role === "member" ? "secondary" : "default"}>
                      {ROLE_LABELS[member.role]}
                    </Badge>
                  )}
                </td>
                <td className="px-3 py-2 align-middle">
                  {member.state === "suspended" ? (
                    <Badge variant="destructive">Suspended</Badge>
                  ) : (
                    <Badge variant="outline">Active</Badge>
                  )}
                </td>
                <td className="px-3 py-2 text-right">
                  <div className="flex justify-end gap-1">
                    {mayChangeState(me, member) && (
                      <Button
                        variant="ghost"
                        size="xs"
                        disabled={pending}
                        onClick={() =>
                          setStateChange({ member, suspend: member.state !== "suspended" })
                        }
                      >
                        {member.state === "suspended" ? (
                          <>
                            <PlayIcon /> Reinstate
                          </>
                        ) : (
                          <>
                            <PauseIcon /> Suspend
                          </>
                        )}
                      </Button>
                    )}
                    <Button
                      variant="ghost"
                      size="xs"
                      title="Reset password"
                      disabled={pending}
                      onClick={() => onResetPassword(member.user_id)}
                    >
                      <KeyRoundIcon /> Reset
                    </Button>
                    {mayRemove(me, member) && (
                      <Button
                        variant="ghost"
                        size="xs"
                        title="Remove from the team"
                        disabled={pending}
                        onClick={() => setRemoval(member)}
                      >
                        <Trash2Icon /> Remove
                      </Button>
                    )}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {state.invitations.length > 0 && (
        <div className="mt-6">
          <h3 className="mb-2 text-ui font-semibold">Pending invitations</h3>
          <ul className="divide-y divide-border rounded-md border border-border">
            {state.invitations.map((invitation) => (
              <li
                key={invitation.invitation_id}
                className="flex items-center justify-between px-3 py-2 text-ui"
              >
                <span>
                  <span className="font-medium">{invitation.email}</span>{" "}
                  <span className="text-muted-foreground">
                    as {ROLE_LABELS[invitation.role].toLowerCase()} ·{" "}
                    {invitation.state === "expired"
                      ? "expired"
                      : `expires ${new Date(invitation.expires_at * 1000).toLocaleString()}`}
                  </span>
                </span>
                {(isOwner || invitation.role === "member") && (
                  <Button
                    variant="ghost"
                    size="xs"
                    disabled={pending}
                    onClick={() => void run(() => revokeInvitation(invitation.invitation_id))}
                  >
                    <XIcon /> Revoke
                  </Button>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      {(state.removals ?? []).length > 0 && (
        <div className="mt-6" data-removals>
          <h3 className="mb-2 text-ui font-semibold">Removed</h3>
          <ul className="divide-y divide-border rounded-md border border-border">
            {(state.removals ?? []).map((removed) => (
              <li
                key={removed.member_id}
                className="px-3 py-2 text-ui"
                data-removed={removed.user_id}
              >
                <div className="flex items-center justify-between">
                  <span>
                    <span className="font-medium">{removed.user_id}</span>{" "}
                    <span className="text-muted-foreground">
                      removed {new Date(removed.removed_at * 1000).toLocaleString()}
                      {removed.removed_by !== null && ` by ${removed.removed_by}`}
                    </span>
                  </span>
                  <span className="flex gap-1">
                    {removed.thread_count > 0 && (
                      <Button
                        variant="ghost"
                        size="xs"
                        disabled={pending}
                        onClick={() => void showThreads(removed.member_id)}
                      >
                        {threads?.memberId === removed.member_id ? "Hide threads" : "Threads"}
                      </Button>
                    )}
                    {hasPendingStep(removed) && (
                      <Button
                        variant="ghost"
                        size="xs"
                        disabled={pending}
                        onClick={() => void run(() => removeMember(removed.member_id))}
                      >
                        <RefreshCwIcon /> Retry now
                      </Button>
                    )}
                  </span>
                </div>
                <RemovalSteps removal={removed} />
                {threads?.memberId === removed.member_id && (
                  <ul className="mt-1 list-disc pl-6 text-sm" data-removed-threads>
                    {threads.list.map((thread) => (
                      <li key={thread.session_id}>
                        <a className="underline" href={withBasePath(`/c/${thread.session_id}`)}>
                          {thread.title ?? "Untitled thread"}
                        </a>
                      </li>
                    ))}
                  </ul>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* ── Invite ─────────────────────────────────────────────── */}
      <Dialog
        open={showInvite}
        onOpenChange={(open) => {
          if (pending) return;
          setShowInvite(open);
          if (!open) setError(null);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Invite someone to the team</DialogTitle>
            <DialogDescription>
              Enter the email address they sign in with. They are let through this installation's
              sign-in with a one-time code sent to that address, then choose their own username and
              password. The invitation works once.
            </DialogDescription>
          </DialogHeader>
          <label className="flex flex-col gap-1 text-ui">
            Email address
            <Input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              disabled={pending}
              autoComplete="off"
            />
          </label>
          {isOwner && (
            <label className="flex flex-col gap-1 text-ui">
              Role
              <select
                className="rounded-md border border-border bg-background px-2 py-1 text-ui"
                value={inviteRole}
                onChange={(e) => setInviteRole(e.target.value as "admin" | "member")}
                disabled={pending}
              >
                <option value="member">Member</option>
                <option value="admin">Admin</option>
              </select>
            </label>
          )}
          {isOwner && linkable.length > 0 && (
            <label className="flex flex-col gap-1 text-ui">
              Existing account (optional)
              <select
                className="rounded-md border border-border bg-background px-2 py-1 text-ui"
                value={linkAccount}
                onChange={(e) => setLinkAccount(e.target.value)}
                disabled={pending}
              >
                <option value="">A new account they create</option>
                {linkable.map((id) => (
                  <option key={id} value={id}>
                    {id}
                  </option>
                ))}
              </select>
            </label>
          )}
          {error !== null && (
            <div
              role="alert"
              className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-ui text-destructive"
            >
              {error}
            </div>
          )}
          <DialogFooter>
            <Button variant="ghost" onClick={() => setShowInvite(false)} disabled={pending}>
              Cancel
            </Button>
            <Button
              onClick={() => void onInvite()}
              loading={pending}
              disabled={email.trim() === ""}
            >
              Create invitation
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* ── The invitation link, shown once ─────────────────────── */}
      <Dialog
        open={inviteLink !== null}
        onOpenChange={(open) => {
          if (!open) setInviteLink(null);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Invitation for {inviteLink?.email}</DialogTitle>
            <DialogDescription>
              Give this link to {inviteLink?.email}, for example in a message from your own email.
              It is shown only this once and works once, for that address only.
            </DialogDescription>
          </DialogHeader>
          {inviteLink !== null && (
            <Input
              value={inviteLink.url}
              readOnly
              className="font-mono text-sm"
              aria-label="Invitation link"
              onFocus={(e) => e.currentTarget.select()}
            />
          )}
          <DialogFooter>
            <Button onClick={() => setInviteLink(null)}>Done</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* ── Suspend or reinstate ───────────────────────────────── */}
      <Dialog
        open={stateChange !== null}
        onOpenChange={(open) => {
          if (pending) return;
          if (!open) {
            setStateChange(null);
            setError(null);
          }
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {stateChange?.suspend ? "Suspend" : "Reinstate"} {stateChange?.member.user_id}?
            </DialogTitle>
            <DialogDescription>
              {stateChange?.suspend
                ? "They are signed out everywhere at once: their sessions, laptops and running work lose access to Omnigent, open pages close, and their schedules pause. Their threads and computers are kept."
                : "They can sign in again. Every sign-in and machine credential from before the suspension stays ended, so they sign in again in the browser and on each of their laptops, and each cloud computer is kept but must be started again before it reconnects. Their schedules resume."}
            </DialogDescription>
          </DialogHeader>
          {error !== null && (
            <div
              role="alert"
              className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-ui text-destructive"
            >
              {error}
            </div>
          )}
          <DialogFooter>
            <Button variant="ghost" onClick={() => setStateChange(null)} disabled={pending}>
              Cancel
            </Button>
            <Button
              variant={stateChange?.suspend ? "destructive" : "default"}
              loading={pending}
              onClick={() => {
                const change = stateChange;
                if (change === null) return;
                void run(() => setMemberSuspended(change.member.member_id, change.suspend)).then(
                  (done) => {
                    if (done) setStateChange(null);
                  },
                );
              }}
            >
              {stateChange?.suspend ? "Suspend" : "Reinstate"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      {/* ── Remove ─────────────────────────────────────────────── */}
      <Dialog
        open={removal !== null}
        onOpenChange={(open) => {
          if (pending) return;
          if (!open) {
            setRemoval(null);
            setError(null);
          }
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Remove {removal?.user_id} from the team?</DialogTitle>
            <DialogDescription>
              This cannot be undone. Their account is deleted at once: every sign-in, CLI login,
              laptop and runner of theirs stops working, their open pages close, their schedules
              stop, and invitations to their address are revoked. The installation's sign-in stops
              letting them in. Their cloud computers stop and are handed over to be deleted. Their
              threads stay, readable by owners and admins; nothing moves to anyone else. Invitations
              they sent to other people stay pending.
            </DialogDescription>
          </DialogHeader>
          {error !== null && (
            <div
              role="alert"
              className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-ui text-destructive"
            >
              {error}
            </div>
          )}
          <DialogFooter>
            <Button variant="ghost" onClick={() => setRemoval(null)} disabled={pending}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              loading={pending}
              onClick={() => {
                const target = removal;
                if (target === null) return;
                void run(() => removeMember(target.member_id)).then((done) => {
                  if (done) setRemoval(null);
                });
              }}
            >
              Remove
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </section>
  );
}
