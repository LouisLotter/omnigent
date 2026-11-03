// The team section of Settings → Members: what an owner and an admin may do,
// the invitation link shown once, and the suspend confirmation. The server
// enforces every rule; these pin the page to the same rules.

import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TeamPanel } from "./TeamPanel";
import * as teamApi from "@/lib/teamApi";
import type { TeamMemberEntry, TeamState } from "@/lib/teamApi";

vi.mock("@/lib/teamApi", () => ({
  inviteToTeam: vi.fn(),
  revokeInvitation: vi.fn(),
  changeMemberRole: vi.fn(),
  setMemberSuspended: vi.fn(),
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

const owner = member("owner", "owner");
const carol = member("carol", "admin");
const alice = member("alice", "member");
const bob = member("bob", "member", "suspended");

function teamAs(me: TeamMemberEntry): TeamState {
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
  };
}

const onChanged = vi.fn(async () => undefined);

function renderAs(me: TeamMemberEntry, accounts: string[] = []) {
  return render(
    <TeamPanel
      state={teamAs(me)}
      accounts={accounts}
      onChanged={onChanged}
      onResetPassword={vi.fn()}
      onRemove={vi.fn()}
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
});
