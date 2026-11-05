// Tests for the admin MembersPage (invite, password reset, delete user).
//
// Browser e2e is impractical (admin/accounts-gated — would need a second
// authenticated server), so the surface is pinned here by mocking the
// mode-agnostic identity probe (resolveIdentity/getCurrentIsAdmin gate admin),
// accountsApi (listUsers/createInvite/resetUserPassword/deleteUser drive the
// table + actions), and useServerInfo (accounts_enabled toggles the
// manage-vs-read-only surface — the latter is the OIDC case).

import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MembersPage } from "./MembersPage";
import type { AccountListEntry } from "@/lib/accountsApi";
import * as accountsApi from "@/lib/accountsApi";
import * as identity from "@/lib/identity";
import * as teamApi from "@/lib/teamApi";

const mocks = vi.hoisted(() => ({
  accountsEnabled: true,
  loginUrl: null as string | null,
  serverVersion: "0.3.0.dev0" as string | null,
  singleUser: false,
  overviewUrls: {} as Record<string, string>,
}));

vi.mock("@/lib/CapabilitiesContext", () => ({
  useServerInfo: () => ({
    accounts_enabled: mocks.accountsEnabled,
    login_url: mocks.loginUrl,
    server_version: mocks.serverVersion,
    single_user: mocks.singleUser,
    persistent_host_providers: ["cloud"],
    persistent_host_overview_urls: mocks.overviewUrls,
  }),
}));
vi.mock("@/lib/identity", () => ({
  resolveIdentity: vi.fn(),
  getCurrentIsAdmin: vi.fn(),
}));
vi.mock("@/lib/teamApi", () => ({
  getTeam: vi.fn(),
  inviteToTeam: vi.fn(),
  revokeInvitation: vi.fn(),
  changeMemberRole: vi.fn(),
  setMemberSuspended: vi.fn(),
  getMemberHosts: vi.fn(async () => ({ ok: true, hosts: [] })),
  getMemberConnections: vi.fn(async () => ({ ok: true, services: [], providers: [] })),
  getMemberThreads: vi.fn(async () => ({ ok: true, threads: [] })),
}));
vi.mock("@/lib/accountsApi", () => ({
  listUsers: vi.fn(),
  createInvite: vi.fn(),
  resetUserPassword: vi.fn(),
  deleteUser: vi.fn(),
}));

function user(overrides: Partial<AccountListEntry> = {}): AccountListEntry {
  return {
    id: "bob",
    is_admin: false,
    created_at: null,
    last_login_at: null,
    has_password: true,
    ...overrides,
  };
}

function renderPage() {
  return render(
    <MemoryRouter>
      <MembersPage />
    </MemoryRouter>,
  );
}

beforeEach(() => {
  mocks.accountsEnabled = true;
  mocks.loginUrl = null;
  mocks.serverVersion = "0.3.0.dev0";
  mocks.singleUser = false;
  mocks.overviewUrls = {};
  vi.mocked(identity.resolveIdentity).mockResolvedValue("admin");
  vi.mocked(identity.getCurrentIsAdmin).mockReturnValue(true);
  vi.mocked(accountsApi.listUsers).mockResolvedValue([]);
  vi.mocked(teamApi.getTeam).mockResolvedValue(null);
  vi.mocked(accountsApi.createInvite).mockResolvedValue({
    ok: true,
    token: "tok",
    register_url: "https://app.example.com/register?invite=tok",
    expires_at: 9_999_999_999,
    is_admin: false,
  });
  vi.mocked(accountsApi.resetUserPassword).mockResolvedValue({
    ok: true,
    id: "bob",
    new_password: "fresh-pw-123",
  });
  vi.mocked(accountsApi.deleteUser).mockResolvedValue({ ok: true });
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("MembersPage gating", () => {
  it("shows a loading state until the identity probe resolves", () => {
    vi.mocked(identity.resolveIdentity).mockReturnValue(new Promise(() => {})); // never resolves
    renderPage();
    expect(screen.getByText("Loading…")).toBeInTheDocument();
  });

  it("blocks non-admins with a permission message and never lists users", async () => {
    vi.mocked(identity.resolveIdentity).mockResolvedValue("alice");
    vi.mocked(identity.getCurrentIsAdmin).mockReturnValue(false);
    renderPage();
    expect(
      await screen.findByText("You don't have permission to manage members."),
    ).toBeInTheDocument();
    expect(accountsApi.listUsers).not.toHaveBeenCalled();
  });

  it("stays in the loading state for an unauthenticated visitor (resolveIdentity redirects)", async () => {
    // resolveIdentity returns null AND owns the login redirect, so the page
    // just never leaves its loading state — it must NOT list users.
    vi.mocked(identity.resolveIdentity).mockResolvedValue(null);
    renderPage();
    await waitFor(() => expect(identity.getCurrentIsAdmin).not.toHaveBeenCalled());
    expect(accountsApi.listUsers).not.toHaveBeenCalled();
    expect(screen.getByText("Loading…")).toBeInTheDocument();
  });
});

describe("MembersPage table", () => {
  it("renders an empty state when there are no members", async () => {
    renderPage();
    expect(await screen.findByText("No members yet.")).toBeInTheDocument();
  });

  it("lists members with role badges and marks the current admin", async () => {
    vi.mocked(accountsApi.listUsers).mockResolvedValue([
      user({ id: "admin", is_admin: true }),
      user({ id: "bob" }),
    ]);
    renderPage();

    const adminRow = (await screen.findByText("admin")).closest("tr")!;
    expect(within(adminRow).getByText("Admin")).toBeInTheDocument();
    expect(within(adminRow).getByText("(you)")).toBeInTheDocument();

    const bobRow = screen.getByText("bob").closest("tr")!;
    expect(within(bobRow).getByText("Member")).toBeInTheDocument();
  });

  it("disables Remove for the current user and Reset for external (passwordless) users", async () => {
    vi.mocked(accountsApi.listUsers).mockResolvedValue([
      user({ id: "admin", is_admin: true }),
      user({ id: "ext", has_password: false }),
    ]);
    renderPage();

    const adminRow = (await screen.findByText("admin")).closest("tr")!;
    expect(within(adminRow).getByRole("button", { name: /Remove/ })).toBeDisabled();

    const extRow = screen.getByText("ext").closest("tr")!;
    expect(within(extRow).getByRole("button", { name: /Reset/ })).toBeDisabled();
  });
});

describe("MembersPage actions", () => {
  it("resets a user's password and shows the new password once", async () => {
    vi.mocked(accountsApi.listUsers).mockResolvedValue([user({ id: "bob" })]);
    renderPage();

    const bobRow = (await screen.findByText("bob")).closest("tr")!;
    fireEvent.click(within(bobRow).getByRole("button", { name: /Reset/ }));

    await waitFor(() => expect(accountsApi.resetUserPassword).toHaveBeenCalledWith("bob"));
    // The new password renders in a readonly, copyable input.
    expect(await screen.findByDisplayValue("fresh-pw-123")).toBeInTheDocument();
    expect(
      screen.getByText(
        "Give this password to them yourself, for example in a message from your own email. It is shown only once; they can change it in Settings → Account.",
      ),
    ).toBeInTheDocument();
  });

  it("deletes a user through the confirmation dialog and refreshes the list", async () => {
    vi.mocked(accountsApi.listUsers)
      .mockResolvedValueOnce([user({ id: "bob" })])
      .mockResolvedValue([]); // after delete → refresh returns empty
    renderPage();

    const bobRow = (await screen.findByText("bob")).closest("tr")!;
    fireEvent.click(within(bobRow).getByRole("button", { name: /Remove/ }));

    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByText("Remove bob?")).toBeInTheDocument();
    fireEvent.click(within(dialog).getByRole("button", { name: /^Remove$/ }));

    await waitFor(() => expect(accountsApi.deleteUser).toHaveBeenCalledWith("bob"));
    expect(await screen.findByText("No members yet.")).toBeInTheDocument();
  });

  it("creates an invite and surfaces the single-use URL", async () => {
    renderPage();
    await screen.findByText("No members yet.");

    fireEvent.click(screen.getByRole("button", { name: /Invite member/ }));
    const dialog = await screen.findByRole("dialog");
    fireEvent.click(within(dialog).getByRole("button", { name: /Create invite/ }));

    await waitFor(() => expect(accountsApi.createInvite).toHaveBeenCalledWith(false));
    // The single-use invite URL renders in a readonly, copyable input.
    expect(await screen.findByDisplayValue(/register\?invite=tok/)).toBeInTheDocument();
  });
});

describe("MembersPage in plain header/single-user mode", () => {
  beforeEach(() => {
    // Explicit single-user local runtime (single_user marker): no accounts,
    // no IdP. The /auth/users endpoint does not exist, so the page must skip
    // the fetch and show a "not available" message instead.
    mocks.accountsEnabled = false;
    mocks.loginUrl = null;
    mocks.serverVersion = "0.3.0.dev0";
    mocks.singleUser = true;
  });

  it("shows a not-available message and never calls listUsers", async () => {
    renderPage();
    expect(
      await screen.findByText("Member management is not available in single-user mode."),
    ).toBeInTheDocument();
    expect(accountsApi.listUsers).not.toHaveBeenCalled();
  });
});

describe("MembersPage under OIDC (read-only)", () => {
  beforeEach(() => {
    // OIDC: accounts disabled but login_url is non-null (IdP present).
    // The list still renders (admins can see who's provisioned), but every
    // management affordance is gone.
    mocks.accountsEnabled = false;
    mocks.loginUrl = "/auth/login";
  });

  it("lists users but offers no management actions", async () => {
    vi.mocked(accountsApi.listUsers).mockResolvedValue([
      user({ id: "admin", is_admin: true }),
      user({ id: "bob" }),
    ]);
    renderPage();

    // The list renders with role badges.
    const bobRow = (await screen.findByText("bob")).closest("tr")!;
    expect(within(bobRow).getByText("Member")).toBeInTheDocument();

    // No invite button, no per-row Reset/Remove, and a read-only notice.
    expect(screen.queryByRole("button", { name: /Invite member/ })).toBeNull();
    expect(within(bobRow).queryByRole("button", { name: /Reset/ })).toBeNull();
    expect(within(bobRow).queryByRole("button", { name: /Remove/ })).toBeNull();
    expect(screen.getByText(/provisioned automatically on first sign-in/)).toBeInTheDocument();
  });
});

describe("MembersPage with a team", () => {
  it("shows the team's members, roles and states instead of the plain account list", async () => {
    vi.mocked(teamApi.getTeam).mockResolvedValue({
      team: { team_id: "t", name: null },
      me: {
        member_id: "o",
        user_id: "admin",
        role: "owner",
        state: "active",
        created_at: 1,
        updated_at: 1,
      },
      members: [
        {
          member_id: "o",
          user_id: "admin",
          role: "owner",
          state: "active",
          created_at: 1,
          updated_at: 1,
        },
        {
          member_id: "b",
          user_id: "bob",
          role: "member",
          state: "suspended",
          created_at: 1,
          updated_at: 1,
        },
      ],
      invitations: [],
      external_identity: "required",
    });
    vi.mocked(accountsApi.listUsers).mockResolvedValue([user({ id: "bob" })]);
    renderPage();
    expect(await screen.findByRole("heading", { name: "Team" })).toBeTruthy();
    expect(screen.getByText("Suspended")).toBeTruthy();
    expect(screen.getByRole("button", { name: /Invite someone/ })).toBeTruthy();
    expect(screen.queryByRole("button", { name: /Invite member/ })).toBeNull();
  });

  it.each([
    ["on this installation's origin", "/platform/computers", true],
    ["on another origin", "https://elsewhere.example/platform/computers", false],
  ] as const)(
    "links a removal to a computers page %s only if same-origin",
    async (_where, path, linked) => {
      const page = path.startsWith("/") ? `${window.location.origin}${path}` : path;
      mocks.overviewUrls = { cloud: page };
      vi.mocked(teamApi.getTeam).mockResolvedValue({
        team: { team_id: "t", name: null },
        me: {
          member_id: "o",
          user_id: "admin",
          role: "owner",
          state: "active",
          created_at: 1,
          updated_at: 1,
        },
        members: [],
        invitations: [],
        external_identity: "required",
        removals: [
          {
            member_id: "d",
            user_id: "dave",
            removed_at: 1_700_000_000,
            removed_by: "admin",
            thread_count: 0,
            steps: {
              access: "done",
              member_group: "done",
              computers: { state: "done", total: 1, handed_over: 1 },
              devices: { state: "not_used", remaining: 0 },
            },
          },
        ],
      });
      renderPage();
      const entry = await waitFor(() => {
        const found = document.querySelector<HTMLElement>('[data-removed="dave"]');
        if (found === null) throw new Error("no removal entry");
        return found;
      });
      const link = within(entry).queryByRole("link", { name: /computers page/ });
      expect(link?.getAttribute("href") ?? null).toBe(linked ? page : null);
    },
  );
});

describe("MembersPage with support access", () => {
  const entry = (member_id: string, user_id: string, role: "owner" | "admin" | "member") => ({
    member_id,
    user_id,
    role,
    state: "active" as const,
    created_at: 1,
    updated_at: 1,
  });
  const owner = entry("o", "admin", "owner");
  const alice = entry("a", "alice", "member");

  function teamState(support: boolean) {
    return {
      team: { team_id: "t", name: null },
      me: owner,
      members: [owner, alice],
      invitations: [],
      external_identity: "required" as const,
      support_access: support,
    };
  }

  function renderAt(path: string) {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    return render(
      <QueryClientProvider client={client}>
        <MemoryRouter initialEntries={[path]}>
          <Routes>
            <Route path="/settings/:section/:subSection?" element={<MembersPage />} />
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>,
    );
  }

  it("opens a member's page from their row's Manage", async () => {
    vi.mocked(teamApi.getTeam).mockResolvedValue(teamState(true));
    renderAt("/settings/members");
    const row = await waitFor(() => {
      const found = document.querySelector<HTMLElement>('[data-member="alice"]');
      if (found === null) throw new Error("no row");
      return found;
    });
    fireEvent.click(within(row).getByRole("button", { name: /Manage/ }));
    expect(await screen.findByRole("heading", { name: "alice" })).toBeTruthy();
    expect(document.querySelector('[data-member-page="a"]')).not.toBeNull();
    expect(teamApi.getMemberHosts).toHaveBeenCalledWith("a");
    fireEvent.click(screen.getByRole("button", { name: /Members/ }));
    expect(await screen.findByRole("heading", { name: "Team" })).toBeTruthy();
  });

  it("shows a member neither Manage nor anyone's page", async () => {
    vi.mocked(identity.resolveIdentity).mockResolvedValue("alice");
    vi.mocked(identity.getCurrentIsAdmin).mockReturnValue(false);
    vi.mocked(teamApi.getTeam).mockResolvedValue(teamState(true));
    renderAt("/settings/members/a");
    expect(
      await screen.findByText("You don't have permission to manage members."),
    ).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Manage/ })).toBeNull();
    expect(document.querySelector("[data-member-page]")).toBeNull();
    expect(teamApi.getTeam).not.toHaveBeenCalled();
    expect(teamApi.getMemberHosts).not.toHaveBeenCalled();
  });

  it("keeps the list, with no Manage, on a server without support access", async () => {
    vi.mocked(teamApi.getTeam).mockResolvedValue(teamState(false));
    renderAt("/settings/members/a");
    expect(await screen.findByRole("heading", { name: "Team" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: /Manage/ })).toBeNull();
    expect(document.querySelector("[data-member-page]")).toBeNull();
  });
});
