/**
 * A member's account actions, for the Members list's rows and the member's own
 * page: suspend or reinstate, reset password and remove, each with the
 * confirmation that says what happens.
 *
 * They follow the server's rules, which it enforces: an owner acts on anyone
 * but themselves (and never removes the installation's owner); an admin acts on
 * members only. An admin also resets their own password. Hiding a button is only
 * convenience.
 */

import { useState } from "react";
import { KeyRoundIcon, PauseIcon, PlayIcon, Trash2Icon } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { type TeamMemberEntry, removeMember, setMemberSuspended } from "@/lib/teamApi";

export function mayChangeState(me: TeamMemberEntry, target: TeamMemberEntry): boolean {
  if (target.member_id === me.member_id) return false;
  return me.role === "owner" || target.role === "member";
}

export function mayRemove(me: TeamMemberEntry, target: TeamMemberEntry): boolean {
  return mayChangeState(me, target) && target.installation_owner !== true;
}

/** An owner resets anyone's password; an admin a member's and their own. */
export function mayResetPassword(me: TeamMemberEntry, target: TeamMemberEntry): boolean {
  return target.member_id === me.member_id || me.role === "owner" || target.role === "member";
}

function ErrorNote({ error }: { error: string | null }) {
  if (error === null) return null;
  return (
    <div
      role="alert"
      className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-ui text-destructive"
    >
      {error}
    </div>
  );
}

export function MemberAccountActions({
  me,
  member,
  onChanged,
  onResetPassword,
  onRemoved,
}: {
  me: TeamMemberEntry;
  member: TeamMemberEntry;
  /** Reload the team after a change. */
  onChanged: () => Promise<void>;
  onResetPassword: (userId: string) => void;
  /** Called after the member is removed, e.g. to leave their page. */
  onRemoved?: () => void;
}) {
  // Whether the open confirmation suspends (true) or reinstates (false).
  const [stateChange, setStateChange] = useState<boolean | null>(null);
  const [removing, setRemoving] = useState(false);
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

  function close() {
    if (pending) return;
    setStateChange(null);
    setRemoving(false);
    setError(null);
  }

  return (
    <>
      {mayChangeState(me, member) && (
        <Button
          variant="ghost"
          size="xs"
          disabled={pending}
          onClick={() => setStateChange(member.state !== "suspended")}
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
      {mayResetPassword(me, member) && (
        <Button
          variant="ghost"
          size="xs"
          title="Reset password"
          disabled={pending}
          onClick={() => onResetPassword(member.user_id)}
        >
          <KeyRoundIcon /> Reset
        </Button>
      )}
      {mayRemove(me, member) && (
        <Button
          variant="ghost"
          size="xs"
          title="Remove from the team"
          disabled={pending}
          onClick={() => setRemoving(true)}
        >
          <Trash2Icon /> Remove
        </Button>
      )}

      {/* ── Suspend or reinstate ───────────────────────────────── */}
      <Dialog open={stateChange !== null} onOpenChange={(open) => !open && close()}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {stateChange ? "Suspend" : "Reinstate"} {member.user_id}?
            </DialogTitle>
            <DialogDescription>
              {stateChange
                ? "They are signed out everywhere at once: their sessions, laptops and running work lose access to Omnigent, open pages close, and their schedules pause. Their threads and computers are kept."
                : "They can sign in again. Every sign-in and machine credential from before the suspension stays ended, so they sign in again in the browser and on each of their laptops, and each cloud computer is kept but must be started again before it reconnects. Their schedules resume."}
            </DialogDescription>
          </DialogHeader>
          <ErrorNote error={error} />
          <DialogFooter>
            <Button variant="ghost" onClick={close} disabled={pending}>
              Cancel
            </Button>
            <Button
              variant={stateChange ? "destructive" : "default"}
              loading={pending}
              onClick={() => {
                const suspend = stateChange;
                if (suspend === null) return;
                void run(() => setMemberSuspended(member.member_id, suspend)).then((done) => {
                  if (done) setStateChange(null);
                });
              }}
            >
              {stateChange ? "Suspend" : "Reinstate"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* ── Remove ─────────────────────────────────────────────── */}
      <Dialog open={removing} onOpenChange={(open) => !open && close()}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Remove {member.user_id} from the team?</DialogTitle>
            <DialogDescription>
              This cannot be undone. Their account is deleted at once: every sign-in, CLI login,
              laptop and runner of theirs stops working, their open pages close, their schedules
              stop, and invitations to their address are revoked. The installation's sign-in stops
              letting them in. Their cloud computers stop and are handed over to be deleted. Their
              threads stay, readable by owners and admins; nothing moves to anyone else. Invitations
              they sent to other people stay pending.
            </DialogDescription>
          </DialogHeader>
          <ErrorNote error={error} />
          <DialogFooter>
            <Button variant="ghost" onClick={close} disabled={pending}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              loading={pending}
              onClick={() => {
                void run(() => removeMember(member.member_id)).then((done) => {
                  if (!done) return;
                  setRemoving(false);
                  onRemoved?.();
                });
              }}
            >
              Remove
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
