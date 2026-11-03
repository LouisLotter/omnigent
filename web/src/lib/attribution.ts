// Who did what on whose behalf: the attribution the server records with each
// turn and approval, and the plain sentences the chat shows for it.
//
// Every value comes from the server's own records. `null` means unknown and
// is shown as "unknown"; nothing here is a credential or prompt content.

export interface TurnInitiator {
  kind: "person" | "schedule" | "agent" | "terminal";
  user: string | null;
  registration: string | null;
  schedule_id: string | null;
  from_session_id: string | null;
  on_behalf_of: string | null;
  /** Everyone who may have typed it, when authorship is not certain. */
  possible?: string[] | null;
  source: string;
}

export interface TurnHost {
  id: string;
  name: string | null;
  owner: string | null;
  kind: "personal" | "cloud_computer" | "sandbox";
  runner_id: string | null;
}

export interface TurnSubscription {
  subject: string | null;
  provider: string | null;
  source: "computer_login" | "configured_provider" | "unknown";
  login_state: "ready" | "needs_auth" | "unavailable" | "unknown";
  login_state_reported_by: "computer" | null;
  account: string | null;
  verified: boolean;
}

export interface RequestedReported {
  requested: string | null;
  reported: string | null;
}

export interface TurnAttribution {
  v: number;
  initiator: TurnInitiator;
  owner: { user: string | null; registration: string | null };
  host: TurnHost | null;
  subscription: TurnSubscription;
  harness: string | null;
  model: RequestedReported;
  effort: RequestedReported;
  recorded_at: number;
}

/** One answer to an approval, with the grant each requirement was answered under. */
export interface ApprovalAnswer {
  by: string | null;
  action: "accept" | "decline";
  at?: number;
  as?: { requirement: number; role: string | null; kind: string; resource: unknown }[];
}

/** What the server shows about one approval (`approval_view`). */
export interface ApprovalRecord {
  elicitation_id: string;
  session_id: string;
  state: "pending" | "approved" | "declined" | "consumed";
  initiator: string | null;
  owner: string | null;
  asked:
    | (Omit<TurnAttribution, "initiator" | "v" | "recorded_at"> & {
        initiator: string | null;
        contributors: string[];
      })
    | null;
  requirements: { kind: string; resource: unknown }[];
  answers: ApprovalAnswer[];
  decided_by: string | null;
  decision_digest: string;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Accept a server attribution object, or `undefined` for anything else. */
export function parseTurnAttribution(raw: unknown): TurnAttribution | undefined {
  if (!isRecord(raw) || !isRecord(raw.initiator) || typeof raw.initiator.kind !== "string") {
    return undefined;
  }
  return raw as unknown as TurnAttribution;
}

/** Accept a server approval record, or `null` for anything else. */
export function parseApprovalRecord(raw: unknown): ApprovalRecord | null {
  if (!isRecord(raw) || typeof raw.elicitation_id !== "string" || !Array.isArray(raw.answers)) {
    return null;
  }
  return raw as unknown as ApprovalRecord;
}

const PROVIDER_NAMES: Record<string, string> = {
  anthropic: "Claude",
  openai: "OpenAI",
  gemini: "Gemini",
};

function providerName(provider: string | null): string | null {
  if (!provider) return null;
  return PROVIDER_NAMES[provider] ?? provider;
}

function person(name: string | null, viewer: string | null, capital = true): string {
  if (!name) return capital ? "Someone" : "someone";
  if (viewer !== null && name === viewer) return capital ? "You" : "you";
  return name;
}

function possessive(name: string | null, viewer: string | null): string {
  if (!name) return "an unknown person's";
  if (viewer !== null && name === viewer) return "your";
  return `${name}'s`;
}

function capitalize(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function initiatorPhrase(initiator: TurnInitiator, viewer: string | null): string {
  switch (initiator.kind) {
    case "schedule":
      // Only the generic words are capitalized; a username is shown as is.
      return `${initiator.user && initiator.user !== viewer ? possessive(initiator.user, viewer) : capitalize(possessive(initiator.user, viewer))} schedule`;
    case "agent":
      return `An agent for ${person(initiator.on_behalf_of, viewer, false)}`;
    case "terminal": {
      const possible = initiator.possible ?? [];
      if (!initiator.user && possible.length > 1) {
        const names = possible.map((name) => person(name, viewer, false)).join(" or ");
        return `${names} in the terminal (not certain)`;
      }
      return `${person(initiator.user, viewer)} in the terminal`;
    }
    default:
      return person(initiator.user, viewer);
  }
}

function subscriptionPhrase(attribution: TurnAttribution, viewer: string | null): string {
  const sub = attribution.subscription;
  const provider = providerName(sub.provider);
  if (sub.source === "configured_provider") {
    return provider ? `the configured ${provider} provider` : "the configured provider";
  }
  if (sub.source === "computer_login") {
    const login = provider ? `${provider} subscription` : "harness login";
    return `${possessive(sub.subject, viewer)} ${login}`;
  }
  return "an unknown subscription";
}

/** The person answerable for the turn: an agent's principal, else its initiator. */
export function responsiblePerson(attribution: TurnAttribution): string | null {
  const { initiator } = attribution;
  return initiator.kind === "agent" ? initiator.on_behalf_of : initiator.user;
}

export interface AttributionSummary {
  /** One plain sentence, e.g. "Bob, on Alice's computer and Claude subscription". */
  text: string;
  /** The full record, one fact per line, for a tooltip. */
  detail: string[];
  /**
   * Shown in the transcript rather than only on hover: someone other than
   * the computer's owner started it, or a schedule or an agent did.
   */
  prominent: boolean;
}

function settingLine(label: string, setting: RequestedReported): string {
  const requested = setting.requested ?? "default";
  const reported = setting.reported ?? "unknown";
  return `${label}: ${reported} (requested ${requested})`;
}

/** Summarize a turn's attribution for *viewer* (their username, or `null`). */
export function summarizeAttribution(
  attribution: TurnAttribution,
  viewer: string | null,
): AttributionSummary {
  const { initiator, host, subscription } = attribution;
  const computer = host ? `${possessive(host.owner, viewer)} computer` : "an unknown computer";
  const text = `${initiatorPhrase(initiator, viewer)}, on ${computer} and ${subscriptionPhrase(attribution, viewer)}`;
  const machineOwner = host?.owner ?? attribution.owner.user;
  const prominent =
    initiator.kind === "schedule" ||
    initiator.kind === "agent" ||
    (initiator.user ?? null) !== (machineOwner ?? null);
  const assumed =
    initiator.source === "owner_assumed"
      ? " (assumed: no typing reached the server)"
      : initiator.source === "terminal_ambiguous"
        ? " (each counts as having taken part)"
        : "";
  const login =
    subscription.login_state_reported_by === "computer"
      ? `; login reported by the computer: ${subscription.login_state.replace("_", " ")}`
      : "";
  const detail = [
    `Started by: ${initiatorPhrase(initiator, null)}${assumed}`,
    `Thread owner: ${attribution.owner.user ?? "unknown"}`,
    host
      ? `Computer: ${host.name ?? host.id} (${host.owner ?? "unknown"}'s ${host.kind.replace("_", " ")})`
      : "Computer: unknown",
    subscription.source === "computer_login"
      ? `Subscription: ${subscription.subject ?? "unknown"}'s configured ${providerName(subscription.provider) ?? "harness"} login (provider account not verified)${login}`
      : `Subscription: ${subscriptionPhrase(attribution, null)}`,
    `Harness: ${attribution.harness ?? "unknown"}`,
    settingLine("Model", attribution.model),
    settingLine("Effort", attribution.effort),
  ];
  return { text, detail, prominent };
}

const ROLE_NAMES: Record<string, string> = {
  thread_owner: "the thread's owner",
  thread_approver: "an approver for this thread",
  service_approver: "an approver for the service",
  team_approver: "a team approver",
};

/** Plain sentences for an approval: who asked, and who answered under which grant. */
export function summarizeApproval(record: ApprovalRecord, viewer: string | null): string[] {
  const lines: string[] = [];
  const asked = record.asked;
  if (asked) {
    const computer = asked.host
      ? `${possessive(asked.host.owner, viewer)} computer`
      : "an unknown computer";
    const joined = asked.contributors.length
      ? ` (${asked.contributors.map((name) => person(name, viewer)).join(", ")} also sent input to this turn)`
      : "";
    lines.push(`Asked during ${possessive(asked.initiator, viewer)} turn, on ${computer}${joined}`);
  }
  for (const answer of record.answers) {
    const verb = answer.action === "decline" ? "Declined" : "Approved";
    const roles = (answer.as ?? [])
      .map((grant) => (grant.role ? (ROLE_NAMES[grant.role] ?? grant.role) : null))
      .filter((role): role is string => role !== null);
    lines.push(
      `${verb} by ${person(answer.by, viewer, false)}${roles.length ? ` as ${roles.join(" and ")}` : ""}`,
    );
  }
  return lines;
}
