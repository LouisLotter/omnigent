/**
 * The team section of Settings → Members: members with their roles and
 * states, pending invitations, and the owner's and admins' actions.
 *
 * Who may do what mirrors the server, which enforces it: an owner invites
 * admins and members, changes roles and suspends or reinstates anyone but
 * themselves; an admin invites members and suspends or reinstates members.
 * Hiding a button is only convenience.
 */

import { useState } from "react";
import { KeyRoundIcon, PauseIcon, PlayIcon, Trash2Icon, UserPlusIcon, XIcon } from "lucide-react";
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
  type TeamMemberEntry,
  type TeamRole,
  type TeamState,
  changeMemberRole,
  inviteToTeam,
  revokeInvitation,
  setMemberSuspended,
} from "@/lib/teamApi";

interface TeamPanelProps {
  state: TeamState;
  /** Reload the team after a change. */
  onChanged: () => Promise<void>;
  onResetPassword: (userId: string) => void;
  onRemove: (userId: string) => void;
  /** Renders an invite link for the current origin and base path. */
  rebaseUrl: (url: string) => string;
}

const ROLE_LABELS: Record<TeamRole, string> = { owner: "Owner", admin: "Admin", member: "Member" };

function mayChangeState(me: TeamMemberEntry, target: TeamMemberEntry): boolean {
  if (target.member_id === me.member_id) return false;
  return me.role === "owner" || target.role === "member";
}

export function TeamPanel({
  state,
  onChanged,
  onResetPassword,
  onRemove,
  rebaseUrl,
}: TeamPanelProps) {
  const { me } = state;
  const isOwner = me.role === "owner";
  const [showInvite, setShowInvite] = useState(false);
  const [email, setEmail] = useState("");
  const [inviteRole, setInviteRole] = useState<"admin" | "member">("member");
  const [inviteLink, setInviteLink] = useState<{ url: string; email: string } | null>(null);
  const [stateChange, setStateChange] = useState<{
    member: TeamMemberEntry;
    suspend: boolean;
  } | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
    const result = await inviteToTeam(email, inviteRole);
    setPending(false);
    if (!result.ok) {
      setError(result.error);
      return;
    }
    setShowInvite(false);
    setInviteLink({ url: rebaseUrl(result.invite_url), email: result.invitation.email });
    setEmail("");
    setInviteRole("member");
    await onChanged();
  }

  const activeMembers = state.members.filter((m) => m.state !== "removed");

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

      {error !== null && !showInvite && stateChange === null && (
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
                    {member.member_id !== me.member_id && (
                      <Button
                        variant="ghost"
                        size="xs"
                        title="Remove user"
                        disabled={pending}
                        onClick={() => onRemove(member.user_id)}
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
                : "They can sign in again. Every sign-in from before the suspension stays ended, so they sign in again in the browser and on each of their laptops; their cloud computers reconnect on their own, and their schedules resume."}
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
    </section>
  );
}
