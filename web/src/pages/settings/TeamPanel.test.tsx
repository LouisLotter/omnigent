// The team section of Settings → Members: what an owner and an admin may do,
// the invitation link shown once, the suspend and remove confirmations, and
// what a removal says is done and what isn't. The server
// enforces every rule; these pin the page to the same rules.

import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TeamPanel } from "./TeamPanel";
import * as teamApi from "@/lib/teamApi";
import type { TeamMemberEntry, TeamRemovalEntry, TeamState } from "@/lib/teamApi";

vi.mock("@/lib/teamApi", () => ({
  inviteToTeam: vi.fn(),
  revokeInvitation: vi.fn(),
  changeMemberRole: vi.fn(),
  setMemberSuspended: vi.fn(),
  removeMember: vi.fn(),
  getRemoval: vi.fn(),
}));

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

const owner = { ...member("owner", "owner"), installation_owner: true };
const carol = member("carol", "admin");
const alice = member("alice", "member");
const bob = member("bob", "member", "suspended");

function removed(
  user_id: string,
  steps: Partial<TeamRemovalEntry["steps"]> = {},
  thread_count = 2,
): TeamRemovalEntry {
  return {
    member_id: `${user_id}-id`,
    user_id,
    removed_at: 1_700_000_000,
    removed_by: "owner",
    thread_count,
    steps: {
      access: "done",
      member_group: "done",
      computers: {
        state: "done",
        total: 1,
        handed_over: 1,
        fleet_url: "https://team.example/platform/computers",
      },
      devices: { state: "not_used", remaining: 0 },
      ...steps,
    },
  };
}

function teamAs(me: TeamMemberEntry, removals: TeamRemovalEntry[] = []): TeamState {
  return {
    team: { team_id: "t", name: null },
    me,
    members: [owner, carol, alice, bob],
    invitations: [
      {
        invitation_id: "inv-admin",
        email: "dave@example.test",
        role: "admin",
        user_id: null,
        state: "pending",
        created_at: 1,
        expires_at: 9_999_999_999,
      },
    ],
    external_identity: "required",
    removals,
  };
}

const onChanged = vi.fn(async () => undefined);

function renderAs(me: TeamMemberEntry, accounts: string[] = [], removals: TeamRemovalEntry[] = []) {
  return render(
    <TeamPanel
      state={teamAs(me, removals)}
      accounts={accounts}
      onChanged={onChanged}
      onResetPassword={vi.fn()}
      rebaseUrl={(url) => url.replace("http://server", "https://team.example")}
    />,
  );
}

function row(name: string): HTMLElement {
  const found = document.querySelector<HTMLElement>(`[data-member="${name}"]`);
  if (found === null) throw new Error(`no row for ${name}`);
  return found;
}

beforeEach(() => {
  onChanged.mockClear();
  vi.mocked(teamApi.setMemberSuspended).mockResolvedValue({ ok: true, member: alice });
  vi.mocked(teamApi.changeMemberRole).mockResolvedValue({ ok: true, member: alice });
  vi.mocked(teamApi.revokeInvitation).mockResolvedValue({
    ok: true,
    invitation: teamAs(owner).invitations[0],
  });
  vi.mocked(teamApi.removeMember).mockResolvedValue({ ok: true, removal: removed("alice") });
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("TeamPanel", () => {
  it("lets an owner change roles and suspend anyone but themselves", () => {
    renderAs(owner);
    expect(within(row("alice")).getByLabelText("Role of alice")).toBeTruthy();
    expect(within(row("carol")).getByRole("button", { name: /Suspend/ })).toBeTruthy();
    expect(within(row("owner")).queryByRole("button", { name: /Suspend/ })).toBeNull();
    expect(within(row("bob")).getByRole("button", { name: /Reinstate/ })).toBeTruthy();
    expect(screen.getByRole("button", { name: /Revoke/ })).toBeTruthy();
  });

  it("lets an admin suspend members only and change no roles", () => {
    renderAs(carol);
    expect(screen.queryByLabelText("Role of alice")).toBeNull();
    expect(within(row("alice")).getByRole("button", { name: /Suspend/ })).toBeTruthy();
    expect(within(row("owner")).queryByRole("button", { name: /Suspend/ })).toBeNull();
    expect(within(row("carol")).queryByRole("button", { name: /Suspend/ })).toBeNull();
    // An admin's invitation cannot be revoked by an admin.
    expect(screen.queryByRole("button", { name: /Revoke/ })).toBeNull();
  });

  it("suspends only after the confirmation that says what happens", async () => {
    renderAs(owner);
    fireEvent.click(within(row("alice")).getByRole("button", { name: /Suspend/ }));
    const dialog = await screen.findByRole("dialog");
    expect(dialog.textContent).toContain("signed out everywhere");
    expect(teamApi.setMemberSuspended).not.toHaveBeenCalled();
    fireEvent.click(within(dialog).getByRole("button", { name: "Suspend" }));
    await waitFor(() => expect(teamApi.setMemberSuspended).toHaveBeenCalledWith("alice-id", true));
    await waitFor(() => expect(onChanged).toHaveBeenCalled());
  });

  it("explains on reinstatement that every device signs in again", async () => {
    renderAs(owner);
    fireEvent.click(within(row("bob")).getByRole("button", { name: /Reinstate/ }));
    const dialog = await screen.findByRole("dialog");
    expect(dialog.textContent).toContain("on each of their laptops");
    fireEvent.click(within(dialog).getByRole("button", { name: "Reinstate" }));
    await waitFor(() => expect(teamApi.setMemberSuspended).toHaveBeenCalledWith("bob-id", false));
  });

  it("shows the invitation link once, as text, for the invited address", async () => {
    vi.mocked(teamApi.inviteToTeam).mockResolvedValue({
      ok: true,
      invitation: { ...teamAs(owner).invitations[0], email: "erin@example.test", role: "member" },
      invite_url: "http://server/register?invite=tok",
    });
    renderAs(owner);
    fireEvent.click(screen.getByRole("button", { name: /Invite someone/ }));
    const dialog = await screen.findByRole("dialog");
    fireEvent.change(within(dialog).getByLabelText("Email address"), {
      target: { value: "erin@example.test" },
    });
    fireEvent.change(within(dialog).getByLabelText("Role"), { target: { value: "admin" } });
    fireEvent.click(within(dialog).getByRole("button", { name: "Create invitation" }));
    await waitFor(() =>
      expect(teamApi.inviteToTeam).toHaveBeenCalledWith("erin@example.test", "admin"),
    );
    const link = (await screen.findByLabelText("Invitation link")) as HTMLInputElement;
    expect(link.value).toBe("https://team.example/register?invite=tok");
  });

  it("lets an owner link an existing account that has no membership", async () => {
    vi.mocked(teamApi.inviteToTeam).mockResolvedValue({
      ok: true,
      invitation: { ...teamAs(owner).invitations[0], email: "frank@example.test", role: "member" },
      invite_url: "http://server/register?invite=tok",
    });
    renderAs(owner, ["owner", "carol", "alice", "bob", "frank"]);
    fireEvent.click(screen.getByRole("button", { name: /Invite someone/ }));
    const dialog = await screen.findByRole("dialog");
    const picker = within(dialog).getByLabelText(
      "Existing account (optional)",
    ) as HTMLSelectElement;
    // Only accounts without a membership are offered.
    expect(Array.from(picker.options).map((option) => option.value)).toEqual(["", "frank"]);
    fireEvent.change(within(dialog).getByLabelText("Email address"), {
      target: { value: "frank@example.test" },
    });
    fireEvent.change(picker, { target: { value: "frank" } });
    fireEvent.click(within(dialog).getByRole("button", { name: "Create invitation" }));
    await waitFor(() =>
      expect(teamApi.inviteToTeam).toHaveBeenCalledWith("frank@example.test", "member", "frank"),
    );
  });

  it("offers an admin no existing account to link", async () => {
    renderAs(carol, ["owner", "carol", "alice", "bob", "frank"]);
    fireEvent.click(screen.getByRole("button", { name: /Invite someone/ }));
    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).queryByLabelText("Existing account (optional)")).toBeNull();
  });

  it("offers an admin no admin role and shows the server's refusal", async () => {
    vi.mocked(teamApi.inviteToTeam).mockResolvedValue({
      ok: false,
      error: "enter a valid email address",
      status: 400,
    });
    renderAs(carol);
    fireEvent.click(screen.getByRole("button", { name: /Invite someone/ }));
    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).queryByLabelText("Role")).toBeNull();
    fireEvent.change(within(dialog).getByLabelText("Email address"), {
      target: { value: "nope" },
    });
    fireEvent.click(within(dialog).getByRole("button", { name: "Create invitation" }));
    expect((await within(dialog).findByRole("alert")).textContent).toContain("valid email");
    expect(teamApi.inviteToTeam).toHaveBeenCalledWith("nope", "member");
  });
  it("lets an owner remove anyone but themselves and the installation's owner", () => {
    const second = member("sam", "owner");
    render(
      <TeamPanel
        state={{ ...teamAs(second), members: [owner, second, carol, alice] }}
        onChanged={onChanged}
        onResetPassword={vi.fn()}
        rebaseUrl={(url) => url}
      />,
    );
    expect(within(row("carol")).getByRole("button", { name: /Remove/ })).toBeTruthy();
    expect(within(row("alice")).getByRole("button", { name: /Remove/ })).toBeTruthy();
    expect(within(row("sam")).queryByRole("button", { name: /Remove/ })).toBeNull();
    expect(within(row("owner")).queryByRole("button", { name: /Remove/ })).toBeNull();
  });

  it("lets an admin remove members only", () => {
    renderAs(carol);
    expect(within(row("alice")).getByRole("button", { name: /Remove/ })).toBeTruthy();
    expect(within(row("owner")).queryByRole("button", { name: /Remove/ })).toBeNull();
    expect(within(row("carol")).queryByRole("button", { name: /Remove/ })).toBeNull();
  });

  it("removes only after the confirmation that says what happens and what stays", async () => {
    renderAs(owner);
    fireEvent.click(within(row("alice")).getByRole("button", { name: /Remove/ }));
    const dialog = await screen.findByRole("dialog");
    expect(dialog.textContent).toContain("cannot be undone");
    expect(dialog.textContent).toContain("readable by owners and admins");
    expect(dialog.textContent).toContain("Invitations they sent to other people stay pending");
    expect(teamApi.removeMember).not.toHaveBeenCalled();
    fireEvent.click(within(dialog).getByRole("button", { name: "Remove" }));
    await waitFor(() => expect(teamApi.removeMember).toHaveBeenCalledWith("alice-id"));
    await waitFor(() => expect(onChanged).toHaveBeenCalled());
  });

  it("shows the server's refusal in the remove confirmation", async () => {
    vi.mocked(teamApi.removeMember).mockResolvedValue({
      ok: false,
      error: "the installation's owner cannot be removed",
      status: 403,
    });
    renderAs(owner);
    fireEvent.click(within(row("carol")).getByRole("button", { name: /Remove/ }));
    const dialog = await screen.findByRole("dialog");
    fireEvent.click(within(dialog).getByRole("button", { name: "Remove" }));
    expect((await within(dialog).findByRole("alert")).textContent).toContain("cannot be removed");
  });

  it("says what a finished removal did, with no retry", () => {
    renderAs(owner, [], [removed("dave")]);
    const entry = document.querySelector<HTMLElement>('[data-removed="dave"]');
    if (entry === null) throw new Error("no removal entry");
    expect(entry.textContent).toContain("Access blocked");
    expect(entry.textContent).toContain("no longer lets them in");
    expect(entry.textContent).toContain("1 of 1");
    expect(entry.textContent).toContain("none to revoke");
    expect(entry.textContent).toContain("2 threads kept");
    const fleet = within(entry).getByRole("link", { name: /computers page/ });
    expect(fleet.getAttribute("href")).toBe("https://team.example/platform/computers");
    expect(within(entry).queryByRole("button", { name: /Retry/ })).toBeNull();
  });

  it("keeps an unfinished step visible and retries it on request", async () => {
    vi.mocked(teamApi.removeMember).mockResolvedValue({ ok: true, removal: removed("dave") });
    renderAs(
      owner,
      [],
      [
        removed("dave", {
          member_group: "pending",
          computers: { state: "pending", total: 1, handed_over: 0, fleet_url: null },
        }),
      ],
    );
    const entry = document.querySelector<HTMLElement>('[data-removed="dave"]');
    if (entry === null) throw new Error("no removal entry");
    expect(entry.textContent).toContain("0 of 1");
    expect(entry.textContent).toContain("retried until it is confirmed");
    fireEvent.click(within(entry).getByRole("button", { name: /Retry now/ }));
    await waitFor(() => expect(teamApi.removeMember).toHaveBeenCalledWith("dave-id"));
    await waitFor(() => expect(onChanged).toHaveBeenCalled());
  });

  it("lists a removed member's threads as links", async () => {
    vi.mocked(teamApi.getRemoval).mockResolvedValue({
      ok: true,
      removal: removed("dave"),
      threads: [
        { session_id: "t1", title: "Quarterly report" },
        { session_id: "t2", title: null },
      ],
    });
    renderAs(owner, [], [removed("dave")]);
    const entry = document.querySelector<HTMLElement>('[data-removed="dave"]');
    if (entry === null) throw new Error("no removal entry");
    fireEvent.click(within(entry).getByRole("button", { name: "Threads" }));
    const report = await within(entry).findByRole("link", { name: "Quarterly report" });
    expect(report.getAttribute("href")).toMatch(/\/c\/t1$/);
    expect(within(entry).getByRole("link", { name: "Untitled thread" })).toBeTruthy();
    expect(teamApi.getRemoval).toHaveBeenCalledWith("dave-id");
  });
});
