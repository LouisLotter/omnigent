// One member's page for owners and admins: their computers with Manage, Wake,
// Delete and Add for them; their connections with Disconnect and no connect;
// their threads; and the account actions under the team's rules. The server
// enforces every rule; these pin the page to them.

import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Host } from "@/hooks/useHosts";
import { authenticatedFetch } from "@/lib/identity";
import * as teamApi from "@/lib/teamApi";
import type { MemberConnections, TeamMemberEntry, TeamState } from "@/lib/teamApi";
import { computerHost } from "@/test/hostFixtures";
import { MemberPage } from "./MemberPage";

vi.mock("@/lib/identity", () => ({ authenticatedFetch: vi.fn() }));
vi.mock("@/lib/teamApi", () => ({
  getMemberHosts: vi.fn(),
  getMemberConnections: vi.fn(),
  getMemberThreads: vi.fn(),
  disconnectMemberProvider: vi.fn(),
  disconnectMemberService: vi.fn(),
  setMemberSuspended: vi.fn(),
  removeMember: vi.fn(),
}));

const fetchMock = vi.mocked(authenticatedFetch);
const response = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

function member(user_id: string, role: TeamMemberEntry["role"], state = "active"): TeamMemberEntry {
  return {
    member_id: `${user_id}-id`,
    user_id,
    role,
    state: state as TeamMemberEntry["state"],
    created_at: 1,
    updated_at: 1,
  };
}

const owner = member("owner", "owner");
const carol = member("carol", "admin");
const alice = member("alice", "member");

function team(me: TeamMemberEntry, members: TeamMemberEntry[] = [owner, carol, alice]): TeamState {
  return {
    team: { team_id: "t", name: null },
    me,
    members,
    invitations: [],
    external_identity: "required",
    support_access: true,
  };
}

const OVERVIEW = "https://overview.example/platform/computers";
let hosts: Host[];
let connections: MemberConnections;
const onChanged = vi.fn(async () => undefined);
const onResetPassword = vi.fn();
const onBack = vi.fn();

function renderPage(me = carol, memberId = "alice-id", members?: TeamMemberEntry[]) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <MemberPage
        memberId={memberId}
        state={team(me, members)}
        providers={["cloudflare"]}
        computersPage={OVERVIEW}
        onChanged={onChanged}
        onResetPassword={onResetPassword}
        onBack={onBack}
      />
    </QueryClientProvider>,
  );
}

function requests(method: string, path: string) {
  return fetchMock.mock.calls.filter(
    ([url, init]) => url === path && (init?.method ?? "GET") === method,
  );
}

beforeEach(() => {
  hosts = [];
  connections = { services: [], providers: [] };
  vi.mocked(teamApi.getMemberHosts).mockImplementation(async () => ({ ok: true, hosts }));
  vi.mocked(teamApi.getMemberConnections).mockImplementation(async () => ({
    ok: true,
    ...connections,
  }));
  vi.mocked(teamApi.getMemberThreads).mockResolvedValue({ ok: true, threads: [] });
  vi.mocked(teamApi.disconnectMemberProvider).mockResolvedValue({ ok: true, disconnected: true });
  vi.mocked(teamApi.disconnectMemberService).mockResolvedValue({ ok: true, disconnected: true });
  fetchMock.mockReset();
  fetchMock.mockImplementation(async (input, init) => {
    const method = init?.method ?? "GET";
    if (input === "/v1/hosts" && method === "POST") {
      return response(computerHost({ state: "waking" }, { host_id: "host_new" }), 201);
    }
    if (String(input).endsWith("/start") && method === "POST") return response({}, 202);
    if (method === "DELETE") return response({ deleted: true });
    throw new Error(`unexpected ${method} ${String(input)}`);
  });
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("MemberPage", () => {
  it("lists the member's computers, each with Manage, Wake or Start again, and Delete", async () => {
    const own = `${window.location.origin}/platform/computers/sb-1`;
    hosts = [
      computerHost({ state: "asleep", manage_url: own }, { host_id: "h1", name: "work" }),
      computerHost({ state: "failed", failure: "No room" }, { host_id: "h2", name: "spare" }),
      computerHost(
        { state: "awake", manage_url: "https://elsewhere.example/platform/computers/x" },
        { host_id: "h3", name: "busy" },
      ),
      { host_id: "laptop", name: "laptop", owner: "alice", status: "online" },
    ];
    renderPage();
    const work = await screen.findByTestId("member-computer-h1");
    expect(screen.queryByTestId("member-computer-laptop")).toBeNull();
    const manage = (row: HTMLElement) =>
      within(row).getByTestId("member-computer-manage").getAttribute("href");
    // Its own page when Omnigent knows it on this origin, else the overview.
    expect(manage(work)).toBe(own);
    expect(manage(screen.getByTestId("member-computer-h2"))).toBe(OVERVIEW);
    expect(manage(screen.getByTestId("member-computer-h3"))).toBe(OVERVIEW);
    expect(
      within(screen.getByTestId("member-computer-h3")).queryByRole("button", { name: "Wake" }),
    ).toBeNull();
    expect(
      within(screen.getByTestId("member-computer-h2")).getByRole("button", { name: "Start again" }),
    ).toBeTruthy();

    fireEvent.click(within(work).getByRole("button", { name: "Wake" }));
    await waitFor(() => expect(requests("POST", "/v1/hosts/h1/start")).toHaveLength(1));

    fireEvent.click(within(work).getByRole("button", { name: "Delete" }));
    const dialog = await screen.findByTestId("delete-computer-dialog");
    expect(within(dialog).getByTestId("delete-computer-sign-out").textContent).toBe(
      "The sign-ins on it are alice's. Ask alice to sign out of them first.",
    );
    fireEvent.click(within(dialog).getByTestId("delete-computer-confirm"));
    await waitFor(() => expect(requests("DELETE", "/v1/hosts/h1")).toHaveLength(1));
  });

  it("adds a computer for the member, which is theirs", async () => {
    renderPage();
    expect(await screen.findByTestId("member-computers-empty")).toBeTruthy();
    fireEvent.click(screen.getByTestId("member-add-computer"));
    const dialog = await screen.findByTestId("add-computer-dialog");
    expect(within(dialog).getByText("Add a computer for alice")).toBeTruthy();
    fireEvent.change(within(dialog).getByTestId("add-computer-name"), {
      target: { value: "Alice's computer" },
    });
    fireEvent.click(within(dialog).getByTestId("add-computer-submit"));
    await waitFor(() => expect(requests("POST", "/v1/hosts")).toHaveLength(1));
    const [, init] = requests("POST", "/v1/hosts")[0];
    expect(JSON.parse(String(init?.body))).toEqual({
      name: "Alice's computer",
      provider: "cloudflare",
      owner_user_id: "alice",
    });
  });

  it("offers Delete but not Wake for a suspended member, and says why", async () => {
    hosts = [computerHost({ state: "asleep" }, { host_id: "h1" })];
    renderPage(carol, "alice-id", [owner, carol, member("alice", "member", "suspended")]);
    const row = await screen.findByTestId("member-computer-h1");
    expect(within(row).queryByRole("button", { name: "Wake" })).toBeNull();
    expect(within(row).getByRole("button", { name: "Delete" })).toBeTruthy();
    expect(screen.getByTestId("member-suspended-note").textContent).toBe(
      "alice is suspended, so their computers can't connect. You can delete them; reinstate alice to use them again.",
    );
    expect(screen.queryByTestId("member-add-computer")).toBeNull();
  });

  it("shows whether the member is connected and disconnects on request, never connects", async () => {
    connections = {
      providers: [{ provider: "github", connected: true, connected_at: 1 }],
      services: [
        {
          id: "tracker",
          title: "Tracker",
          auth: "bearer",
          connected: true,
          needs_reconnect: true,
          disconnectable: true,
        },
        {
          id: "notes",
          title: "Notes",
          auth: "oauth",
          connected: false,
          needs_reconnect: false,
          disconnectable: false,
        },
      ],
    };
    renderPage();
    const github = await waitFor(() => {
      const found = document.querySelector<HTMLElement>('[data-connection="provider:github"]');
      if (found === null) throw new Error("no GitHub row");
      return found;
    });
    expect(github.textContent).toContain("GitHub · Connected");
    const tracker = document.querySelector<HTMLElement>('[data-connection="mcp:tracker"]');
    const notes = document.querySelector<HTMLElement>('[data-connection="mcp:notes"]');
    expect(tracker?.textContent).toContain("Needs reconnecting");
    expect(notes?.textContent).toContain("Not connected");
    expect(within(notes as HTMLElement).queryByRole("button")).toBeNull();
    expect(screen.queryByRole("button", { name: /Connect/ })).toBeNull();
    expect(screen.getByTestId("member-connections-note").textContent).toBe(
      "alice connects these themselves, from their own Settings → MCP and Settings → Sandbox Integrations.",
    );

    fireEvent.click(within(github).getByRole("button", { name: "Disconnect" }));
    await waitFor(() =>
      expect(teamApi.disconnectMemberProvider).toHaveBeenCalledWith("alice-id", "github"),
    );
    fireEvent.click(within(tracker as HTMLElement).getByRole("button", { name: "Disconnect" }));
    await waitFor(() =>
      expect(teamApi.disconnectMemberService).toHaveBeenCalledWith("alice-id", "tracker"),
    );
    expect(teamApi.getMemberConnections).toHaveBeenCalledTimes(3);
  });

  it("lists the member's threads as links", async () => {
    vi.mocked(teamApi.getMemberThreads).mockResolvedValue({
      ok: true,
      threads: [
        { session_id: "t1", title: "Quarterly report" },
        { session_id: "t2", title: null },
      ],
    });
    renderPage();
    const report = await screen.findByRole("link", { name: "Quarterly report" });
    expect(report.getAttribute("href")).toMatch(/\/c\/t1$/);
    expect(screen.getByRole("link", { name: "Untitled thread" })).toBeTruthy();
    expect(teamApi.getMemberThreads).toHaveBeenCalledWith("alice-id");
  });

  it("offers the account actions the team's rules allow", async () => {
    renderPage(carol, "alice-id");
    const account = await screen.findByTestId("member-account");
    for (const name of [/Suspend/, /Reset/, /Remove/]) {
      expect(within(account).getByRole("button", { name })).toBeTruthy();
    }
    fireEvent.click(within(account).getByRole("button", { name: /Reset/ }));
    expect(onResetPassword).toHaveBeenCalledWith("alice");
    cleanup();
    // An admin acts on no owner's account, a reset included.
    renderPage(carol, "owner-id");
    expect((await screen.findByTestId("member-account")).querySelectorAll("button")).toHaveLength(
      0,
    );
  });

  it("says when the person is not on the team", () => {
    renderPage(carol, "gone-id");
    expect(screen.getByTestId("member-missing").textContent).toBe(
      "This person is not on the team.",
    );
    fireEvent.click(screen.getByRole("button", { name: /Members/ }));
    expect(onBack).toHaveBeenCalled();
    expect(teamApi.getMemberHosts).not.toHaveBeenCalled();
  });
});
