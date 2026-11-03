import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ApprovalRecord } from "@/lib/attribution";
import { ApprovalAttribution, useApprovalRecord } from "./ApprovalAttribution";

const record: ApprovalRecord = {
  elicitation_id: "elicit_1",
  session_id: "conv_1",
  state: "consumed",
  initiator: "alice",
  owner: "alice",
  asked: {
    initiator: "alice",
    contributors: ["bob"],
    owner: { user: "alice", registration: null },
    host: { id: "h1", name: "alice-laptop", owner: "alice", kind: "personal", runner_id: null },
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
    model: { requested: null, reported: null },
    effort: { requested: null, reported: null },
  },
  requirements: [{ kind: "team", resource: [2, "tracker"] }],
  answers: [
    {
      by: "carol",
      action: "accept",
      as: [{ requirement: 0, role: "service_approver", kind: "team", resource: [2, "tracker"] }],
    },
  ],
  decided_by: "carol",
  decision_digest: "8ba2",
};

function Probe({ status }: { status: string }) {
  const fetched = useApprovalRecord("conv_1", "elicit_1", status);
  return <ApprovalAttribution record={fetched} />;
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("ApprovalAttribution", () => {
  it("says who asked, whose input joined the turn, and who answered as what", () => {
    render(<ApprovalAttribution record={record} />);
    const lines = screen.getByTestId("approval-attribution");
    expect(lines).toHaveTextContent(
      "Asked during alice's turn, on alice's computer (bob also sent input to this turn)",
    );
    expect(lines).toHaveTextContent("Approved by carol as an approver for the service");
    expect(lines).toHaveAttribute("title", "Decision digest 8ba2");
  });

  it("shows nothing for a prompt that is not a recorded approval", async () => {
    const fetchMock = vi.fn(
      async () => new Response(JSON.stringify({ status: "pending" }), { status: 200 }),
    );
    vi.stubGlobal("fetch", fetchMock);
    render(<Probe status="pending" />);
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    expect(screen.queryByTestId("approval-attribution")).toBeNull();
  });

  it("reads the record from the prompt's endpoint", async () => {
    const fetchMock = vi.fn(
      async (_input: RequestInfo | URL) =>
        new Response(JSON.stringify({ status: "resolved", approval: record }), { status: 200 }),
    );
    vi.stubGlobal("fetch", fetchMock);
    render(<Probe status="responded" />);
    expect(await screen.findByTestId("approval-attribution")).toHaveTextContent(
      "Approved by carol",
    );
    expect(String(fetchMock.mock.calls[0]?.[0])).toContain(
      "/v1/sessions/conv_1/elicitations/elicit_1",
    );
  });
});
