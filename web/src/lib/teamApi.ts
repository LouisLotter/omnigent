/**
 * Client for the team API (``/v1/team``): members, roles, suspension,
 * removal and invitations. Every helper resolves with a typed result instead of
 * throwing, like ``accountsApi``.
 */

import type { Host } from "@/hooks/useHosts";
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
  /** The installation's own owner, whom the team never removes. */
  installation_owner?: boolean;
}

/** One step of a removal: done, still pending (it is retried), or not used here. */
export type RemovalStepState = "done" | "pending" | "not_used";

export interface TeamRemovalEntry {
  member_id: string;
  user_id: string;
  removed_at: number;
  /** The owner or admin who removed them, when known. */
  removed_by: string | null;
  /** Threads they owned; owners and admins can still open them. */
  thread_count: number;
  steps: {
    /** Every sign-in, token, runner and invitation of theirs was refused at once. */
    access: "done";
    /** The installation's sign-in no longer admits them (read back by the proxy). */
    member_group: RemovalStepState;
    computers: {
      state: RemovalStepState;
      total: number;
      handed_over: number;
    };
    devices: { state: RemovalStepState; remaining: number };
  };
}

export interface RemovedThread {
  session_id: string;
  title: string | null;
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
  /** The latest removals, newest first; absent from servers before removals were recorded. */
  removals?: TeamRemovalEntry[];
  /** How the server ties requests to a front proxy's sign-in: off, bind or required. */
  external_identity: "off" | "bind" | "required";
  /**
   * Whether owners and admins manage other members' computers and connections, on each
   * member's page; absent from servers without support access.
   */
  support_access?: boolean;
}

/** A member's connection to a provider such as GitHub; never the login itself. */
export interface MemberProviderConnection {
  provider: string;
  connected: boolean;
  connected_at: number | null;
}

/** Whether a member is connected to one MCP service. */
export interface MemberServiceConnection {
  id: string;
  title: string;
  auth: string;
  connected: boolean;
  needs_reconnect: boolean;
  /** The member keeps their own login to it, which an owner or admin may remove. */
  disconnectable: boolean;
}

export interface MemberConnections {
  services: MemberServiceConnection[];
  providers: MemberProviderConnection[];
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

/**
 * POST /v1/team/members/{id}/remove — removes a member for good. Repeated, it
 * retries the steps not done yet and returns where the removal stands.
 */
export function removeMember(memberId: string): Promise<TeamResult<{ removal: TeamRemovalEntry }>> {
  return teamRequest(`/v1/team/members/${encodeURIComponent(memberId)}/remove`, jsonInit("POST"));
}

/** GET /v1/team/members/{id}/removal — a removal's steps and the threads they owned. */
export function getRemoval(
  memberId: string,
): Promise<TeamResult<{ removal: TeamRemovalEntry; threads: RemovedThread[] }>> {
  return teamRequest(`/v1/team/members/${encodeURIComponent(memberId)}/removal`);
}

function memberPath(memberId: string, suffix: string): string {
  return `/v1/team/members/${encodeURIComponent(memberId)}${suffix}`;
}

/** GET /v1/team/members/{id}/hosts — a member's hosts, for owners and admins. */
export function getMemberHosts(memberId: string): Promise<TeamResult<{ hosts: Host[] }>> {
  return teamRequest(memberPath(memberId, "/hosts"));
}

/** GET /v1/team/members/{id}/connections — whether the member is connected to each service. */
export function getMemberConnections(memberId: string): Promise<TeamResult<MemberConnections>> {
  return teamRequest(memberPath(memberId, "/connections"));
}

/** DELETE /v1/team/members/{id}/connections/{provider} — remove the member's connection. */
export function disconnectMemberProvider(
  memberId: string,
  provider: string,
): Promise<TeamResult<{ disconnected: boolean }>> {
  return teamRequest(
    memberPath(memberId, `/connections/${encodeURIComponent(provider)}`),
    jsonInit("DELETE"),
  );
}

/** DELETE /v1/team/members/{id}/mcp-services/{service}/connection — remove their login. */
export function disconnectMemberService(
  memberId: string,
  serviceId: string,
): Promise<TeamResult<{ disconnected: boolean }>> {
  return teamRequest(
    memberPath(memberId, `/mcp-services/${encodeURIComponent(serviceId)}/connection`),
    jsonInit("DELETE"),
  );
}

/** GET /v1/team/members/{id}/threads — the threads the member owns, or owned. */
export function getMemberThreads(
  memberId: string,
): Promise<TeamResult<{ threads: RemovedThread[] }>> {
  return teamRequest(memberPath(memberId, "/threads"));
}

/** GET /v1/team/invitations/preview — what an invitation link offers; no sign-in needed. */
export async function previewInvitation(token: string): Promise<InvitationPreview | null> {
  const result = await teamRequest<InvitationPreview>(
    `/v1/team/invitations/preview?token=${encodeURIComponent(token)}`,
  );
  return result.ok ? result : null;
}
