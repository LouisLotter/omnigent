import { describe, expect, it } from "vitest";
import {
  parseApprovalRecord,
  parseTurnAttribution,
  summarizeApproval,
  summarizeAttribution,
  type ApprovalRecord,
  type TurnAttribution,
} from "./attribution";

function turn(overrides: Partial<TurnAttribution> = {}): TurnAttribution {
  return {
    v: 1,
    initiator: {
      kind: "person",
      user: "bob",
      registration: null,
      schedule_id: null,
      from_session_id: null,
      on_behalf_of: null,
      source: "signed_in",
    },
    owner: { user: "alice", registration: null },
    host: { id: "h1", name: "alice-laptop", owner: "alice", kind: "personal", runner_id: "r1" },
    subscription: {
      subject: "alice",
      provider: "anthropic",
      source: "computer_login",
      login_state: "ready",
      login_state_reported_by: "computer",
      account: null,
      verified: false,
    },
    harness: "claude-native",
    model: { requested: "claude-opus-4-8", reported: null },
    effort: { requested: "high", reported: "xhigh" },
    recorded_at: 1,
    ...overrides,
  };
}

describe("summarizeAttribution", () => {
  it("names the colleague, the owner's computer and subscription, and shows it plainly", () => {
    const summary = summarizeAttribution(turn(), "alice");
    expect(summary.text).toBe("bob, on your computer and your Claude subscription");
    expect(summary.prominent).toBe(true);
    expect(summarizeAttribution(turn(), "carol").text).toBe(
      "bob, on alice's computer and alice's Claude subscription",
    );
  });

  it("keeps a turn the computer's owner started in the hover row", () => {
    const own = turn({ initiator: { ...turn().initiator, user: "alice" } });
    const summary = summarizeAttribution(own, "alice");
    expect(summary.text).toBe("You, on your computer and your Claude subscription");
    expect(summary.prominent).toBe(false);
  });

  it("shows schedules and agents plainly, with the person they act for", () => {
    const schedule = turn({
      initiator: { ...turn().initiator, kind: "schedule", user: "alice", schedule_id: "t1" },
    });
    expect(summarizeAttribution(schedule, "bob")).toMatchObject({
      text: "alice's schedule, on alice's computer and alice's Claude subscription",
      prominent: true,
    });
    const agent = turn({
      initiator: { ...turn().initiator, kind: "agent", user: null, on_behalf_of: "bob" },
    });
    expect(summarizeAttribution(agent, "bob")).toMatchObject({
      text: "An agent for you, on alice's computer and alice's Claude subscription",
      prominent: true,
    });
  });

  it("says when the subscription's provider account is not verified, and what is unknown", () => {
    const detail = summarizeAttribution(turn(), null).detail;
    expect(detail).toContain(
      "Subscription: alice's configured Claude login (provider account not verified); login reported by the computer: ready",
    );
    expect(detail).toContain("Model: unknown (requested claude-opus-4-8)");
    expect(detail).toContain("Effort: xhigh (requested high)");
  });

  it("marks terminal typing nobody was attached for as assumed", () => {
    const typed = turn({
      initiator: { ...turn().initiator, kind: "terminal", user: "alice", source: "owner_assumed" },
    });
    const summary = summarizeAttribution(typed, "bob");
    expect(summary.text).toBe(
      "alice in the terminal, on alice's computer and alice's Claude subscription",
    );
    expect(summary.prominent).toBe(false);
    expect(summary.detail[0]).toBe(
      "Started by: alice in the terminal (assumed: nobody was attached)",
    );
  });

  it("describes a configured provider and an unknown computer", () => {
    const configured = turn({
      host: null,
      subscription: { ...turn().subscription, source: "configured_provider", subject: null },
    });
    expect(summarizeAttribution(configured, null).text).toBe(
      "bob, on an unknown computer and the configured Claude provider",
    );
  });

  it("parses only attribution-shaped values", () => {
    expect(parseTurnAttribution(turn())).toBeDefined();
    expect(parseTurnAttribution({ initiator: "bob" })).toBeUndefined();
    expect(parseTurnAttribution(null)).toBeUndefined();
  });
});

describe("summarizeApproval", () => {
  const record: ApprovalRecord = {
    elicitation_id: "elicit_1",
    session_id: "s1",
    state: "approved",
    initiator: "alice",
    owner: "alice",
    asked: {
      ...turn(),
      initiator: "alice",
      contributors: ["bob"],
    },
    requirements: [
      { kind: "personal", resource: [1, "s1"] },
      { kind: "team", resource: [3, ""] },
    ],
    answers: [
      {
        by: "alice",
        action: "accept",
        as: [{ requirement: 0, role: "thread_owner", kind: "personal", resource: [1, "s1"] }],
      },
      {
        by: "dave",
        action: "decline",
        as: [{ requirement: 1, role: "team_approver", kind: "team", resource: [3, ""] }],
      },
    ],
    decided_by: "dave",
    decision_digest: "ab",
  };

  it("says who asked, whose input joined the turn and who answered as what", () => {
    expect(summarizeApproval(record, "alice")).toEqual([
      "Asked during your turn, on your computer (bob also sent input to this turn)",
      "Approved by you as the thread's owner",
      "Declined by dave as a team approver",
    ]);
  });

  it("parses only approval-shaped values", () => {
    expect(parseApprovalRecord(record)).toBe(record);
    expect(parseApprovalRecord({ elicitation_id: "x" })).toBeNull();
  });
});
