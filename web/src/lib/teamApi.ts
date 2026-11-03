/**
 * Client for the team API (``/v1/team``): members, roles, suspension and
 * invitations. Every helper resolves with a typed result instead of
 * throwing, like ``accountsApi``.
 */

import { hostFetch } from "./host";

export type TeamRole = "owner" | "admin" | "member";
export type TeamMemberState = "invited" | "active" | "suspended" | "removed";
export type InvitationState = "pending" | "redeemed" | "revoked" | "expired";

export interface TeamMemberEntry {
  member_id: string;
  user_id: string;
  role: TeamRole;
  state: TeamMemberState;
  created_at: number;
  updated_at: number;
}

export interface TeamInvitationEntry {
  invitation_id: string;
  email: string;
  role: "admin" | "member";
  user_id: string | null;
  state: InvitationState;
  created_at: number;
  expires_at: number;
}

export interface TeamState {
  team: { team_id: string; name: string | null } | null;
  me: TeamMemberEntry;
  members: TeamMemberEntry[];
  invitations: TeamInvitationEntry[];
  /** How the server ties requests to a front proxy's sign-in: off, bind or required. */
  external_identity: "off" | "bind" | "required";
}

export interface InvitationPreview {
  team_name: string | null;
  email: string;
  role: "admin" | "member";
  /** The invitation links an existing account, proved by its password at sign-in. */
  links_account: boolean;
  /** That account's username, when it links one. */
  user_id?: string | null;
  expires_at: number;
}

export interface TeamFailure {
  ok: false;
  error: string;
  status: number;
}
export type TeamResult<T> = ({ ok: true } & T) | TeamFailure;

async function teamRequest<T>(path: string, init?: RequestInit): Promise<TeamResult<T>> {
  let res: Response;
  try {
    res = await hostFetch(path, { cache: "no-store", ...init });
  } catch {
    return { ok: false, error: "Could not reach the server.", status: 0 };
  }
  if (res.ok) {
    const body = (await res.json()) as T;
    return { ok: true, ...body };
  }
  let message = `Request failed (${res.status}).`;
  try {
    const data = (await res.json()) as { error?: string | { message?: string } };
    if (typeof data.error === "string") message = data.error;
    else if (data.error?.message) message = data.error.message;
  } catch {
    // Body wasn't JSON; keep the generic message.
  }
  return { ok: false, error: message, status: res.status };
}

function jsonInit(method: string, body?: unknown): RequestInit {
  return body === undefined
    ? { method }
    : { method, headers: { "content-type": "application/json" }, body: JSON.stringify(body) };
}

/** GET /v1/team — owners and admins; ``null`` when the server has no team API or refuses. */
export async function getTeam(): Promise<TeamState | null> {
  const result = await teamRequest<TeamState>("/v1/team");
  return result.ok ? result : null;
}

/**
 * POST /v1/team/invitations — the single-use link is returned once. With
 * *userId* (owners only) it links that existing account instead of creating one.
 */
export function inviteToTeam(
  email: string,
  role: "admin" | "member",
  userId?: string,
): Promise<TeamResult<{ invitation: TeamInvitationEntry; invite_url: string }>> {
  const body = userId ? { email, role, user_id: userId } : { email, role };
  return teamRequest("/v1/team/invitations", jsonInit("POST", body));
}

/** DELETE /v1/team/invitations/{id}. */
export function revokeInvitation(
  invitationId: string,
): Promise<TeamResult<{ invitation: TeamInvitationEntry }>> {
  return teamRequest(
    `/v1/team/invitations/${encodeURIComponent(invitationId)}`,
    jsonInit("DELETE"),
  );
}

/** PUT /v1/team/members/{id}/role — owners only. */
export function changeMemberRole(
  memberId: string,
  role: TeamRole,
): Promise<TeamResult<{ member: TeamMemberEntry }>> {
  return teamRequest(
    `/v1/team/members/${encodeURIComponent(memberId)}/role`,
    jsonInit("PUT", { role }),
  );
}

/** POST /v1/team/members/{id}/suspend or /reinstate. */
export function setMemberSuspended(
  memberId: string,
  suspended: boolean,
): Promise<TeamResult<{ member: TeamMemberEntry }>> {
  const action = suspended ? "suspend" : "reinstate";
  return teamRequest(
    `/v1/team/members/${encodeURIComponent(memberId)}/${action}`,
    jsonInit("POST"),
  );
}

/** GET /v1/team/invitations/preview — what an invitation link offers; no sign-in needed. */
export async function previewInvitation(token: string): Promise<InvitationPreview | null> {
  const result = await teamRequest<InvitationPreview>(
    `/v1/team/invitations/preview?token=${encodeURIComponent(token)}`,
  );
  return result.ok ? result : null;
}
