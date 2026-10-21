/**
 * Presentation rules for persistent managed hosts ("computers"): server-managed
 * hosts that outlive sessions, sleep when idle and wake when work arrives.
 */

import type { Host, PersistentHostState, PersistentHostWakeStage } from "@/hooks/useHosts";
import { sandboxOptionLabel } from "@/lib/capabilities";

const STATE_LABELS: Record<PersistentHostState, string> = {
  awake: "Awake",
  asleep: "Asleep",
  waking: "Waking…",
  sleeping: "Going to sleep",
  failed: "Failed",
};

const WAKE_STAGE_LABELS: Record<PersistentHostWakeStage, string> = {
  provisioning: "Starting the computer",
  starting: "Starting Omnigent",
  connecting: "Connecting",
};

export const SIGN_IN_NEEDED_MESSAGE =
  "Sign-in needed: open Set up / manage to sign in on this computer.";

export function isPersistentHost(host: Host): boolean {
  return host.lifecycle === "persistent";
}

/**
 * Whether work can be sent to *host* now: it is online, or it is a computer
 * that wakes when work arrives (any state but failed). An admission refusal
 * is reported separately by {@link persistentHostRefusal}: it may be stale.
 */
export function isHostAvailable(host: Host): boolean {
  return host.status === "online" || (isPersistentHost(host) && !isPersistentHostFailed(host));
}

/** Whether a computer is waking or going to sleep, so its progress is worth polling. */
export function isPersistentHostChanging(host: Host): boolean {
  const state = host.persistent_status?.state;
  return isPersistentHost(host) && (state === "waking" || state === "sleeping");
}

/** Whether a computer failed to start and can be started again. */
export function isPersistentHostFailed(host: Host): boolean {
  return isPersistentHost(host) && host.persistent_status?.state === "failed";
}

export interface PersistentHostStateDescription {
  /** "Asleep", "Waking…", "Going to sleep", "Failed" or "Awake". */
  label: string;
  /** Wake progress or failure reason, when there is one. */
  detail: string | null;
}

/** A computer's state for display; null for any other host. */
export function describePersistentHostState(host: Host): PersistentHostStateDescription | null {
  const status = host.persistent_status;
  if (!isPersistentHost(host) || !status) return null;
  // A state from a newer server still renders rather than disappearing.
  const label = STATE_LABELS[status.state] ?? status.state;
  if (status.state === "waking" && status.wake_stage) {
    return { label, detail: WAKE_STAGE_LABELS[status.wake_stage] ?? null };
  }
  if (status.state === "failed") return { label, detail: status.failure };
  return { label, detail: null };
}

/**
 * One-line state text, e.g. "Waking… Starting the computer" or
 * "Failed: <reason>"; null for other hosts.
 */
export function persistentHostStateText(host: Host): string | null {
  const description = describePersistentHostState(host);
  if (!description) return null;
  if (!description.detail) return description.label;
  const separator = host.persistent_status?.state === "failed" ? ": " : " ";
  return `${description.label}${separator}${description.detail}`;
}

/**
 * Short status word for compact host labels: the computer's state while it is
 * not awake, else the host's online/offline status.
 */
export function hostStatusWord(host: Host): string {
  const state = host.persistent_status?.state;
  if (isPersistentHost(host) && state && state !== "awake") {
    return (STATE_LABELS[state] ?? state).toLowerCase();
  }
  return host.status;
}

export interface PersistentHostRefusal {
  message: string;
  needsSignIn: boolean;
}

/** Why a computer refuses new work, or null when it admits work or has not said. */
export function persistentHostRefusal(host: Host): PersistentHostRefusal | null {
  const admission = host.persistent_status?.admission;
  if (!isPersistentHost(host) || !admission || admission.admitting) return null;
  if (admission.reason === "needs_sign_in") {
    return { message: SIGN_IN_NEEDED_MESSAGE, needsSignIn: true };
  }
  return {
    message: admission.message ?? "This computer isn't accepting new work right now.",
    needsSignIn: false,
  };
}

/** The computer's provider setup page, when it has one. */
export function persistentHostManageUrl(host: Host): string | null {
  return isPersistentHost(host) ? (host.persistent_status?.manage_url ?? null) : null;
}

/**
 * Name to show for a host: per-session sandboxes read as their provider
 * ("Modal Sandbox"); computers and connected machines by their own name.
 */
export function hostDisplayLabel(host: Host): string {
  return host.sandbox_provider && !isPersistentHost(host)
    ? sandboxOptionLabel(host.sandbox_provider)
    : host.name;
}
