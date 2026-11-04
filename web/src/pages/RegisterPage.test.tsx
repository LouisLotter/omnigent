// Tests for RegisterPage (the /register?invite=… invite-redemption page).
//
// Mirrors LoginPage.test.tsx: window.location.href is stubbed so success
// navigation is observable without jsdom navigating. The page calls
// register() from accountsApi; success hard-navigates to "/", failure shows a
// role="alert". The page also gates on the invite query param (missing → alert,
// no form).

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { RegisterPage } from "./RegisterPage";
import * as accountsApi from "@/lib/accountsApi";
import { CapabilitiesProvider } from "@/lib/CapabilitiesContext";
import { FALLBACK_SERVER_INFO } from "@/lib/capabilities";
import * as teamApi from "@/lib/teamApi";

vi.mock("@/lib/accountsApi", () => ({ register: vi.fn(), login: vi.fn() }));
vi.mock("@/lib/teamApi", () => ({ previewInvitation: vi.fn() }));

const ORIGIN = "https://app.example.com";
let hrefWrites: string[];
let originalLocation: Location;

function renderRegisterAt(search: string) {
  return render(
    <MemoryRouter initialEntries={[`/register${search}`]}>
      <RegisterPage />
    </MemoryRouter>,
  );
}

function fillForm(username: string, password: string, confirm: string) {
  fireEvent.change(screen.getByLabelText(/username/i), { target: { value: username } });
  fireEvent.change(screen.getByLabelText(/^password/i), { target: { value: password } });
  fireEvent.change(screen.getByLabelText(/confirm password/i), { target: { value: confirm } });
}

beforeEach(() => {
  hrefWrites = [];
  originalLocation = window.location;
  Object.defineProperty(window, "location", {
    configurable: true,
    value: {
      origin: ORIGIN,
      set href(value: string) {
        hrefWrites.push(value);
      },
      get href() {
        return hrefWrites[hrefWrites.length - 1] ?? `${ORIGIN}/register`;
      },
    },
  });
  vi.mocked(accountsApi.register).mockResolvedValue({
    ok: true,
    user: { id: "alice", is_admin: false },
    token: "t",
    expires_in: 3600,
  });
  vi.mocked(teamApi.previewInvitation).mockResolvedValue(null);
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  // useLoginV2 persists ?login-v2 to localStorage — clear it so the v2 flag
  // doesn't leak into the v1 tests (which assume the default-off state).
  try {
    window.localStorage.clear();
  } catch {
    /* jsdom localStorage always present, but be safe */
  }
  Object.defineProperty(window, "location", { configurable: true, value: originalLocation });
});

describe("RegisterPage", () => {
  it("shows an invite-required alert and no form when the invite token is missing", () => {
    renderRegisterAt("");
    expect(screen.getByRole("alert")).toHaveTextContent(/invite token/i);
    expect(screen.queryByLabelText(/username/i)).not.toBeInTheDocument();
  });

  it("renders the form when an invite token is present", () => {
    renderRegisterAt("?invite=tok123");
    expect(screen.getByLabelText(/username/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /create account/i })).toBeInTheDocument();
  });

  it("disables submit until a username and an 8+ char password are entered", () => {
    renderRegisterAt("?invite=tok123");
    const submit = screen.getByRole("button", { name: /create account/i });
    expect(submit).toBeDisabled();

    fillForm("alice", "short", "short");
    expect(submit).toBeDisabled(); // password < 8

    fillForm("alice", "longenough", "longenough");
    expect(submit).toBeEnabled();
  });

  it("blocks submit and shows an error when the passwords don't match", () => {
    renderRegisterAt("?invite=tok123");
    fillForm("alice", "longenough", "different1");
    fireEvent.click(screen.getByRole("button", { name: /create account/i }));

    expect(screen.getByRole("alert")).toHaveTextContent("Passwords don't match.");
    expect(accountsApi.register).not.toHaveBeenCalled();
  });

  it("redeems the invite and navigates home on success", async () => {
    renderRegisterAt("?invite=tok123");
    fillForm("alice", "longenough", "longenough");
    fireEvent.click(screen.getByRole("button", { name: /create account/i }));

    await waitFor(() =>
      expect(accountsApi.register).toHaveBeenCalledWith({
        invite: "tok123",
        username: "alice",
        password: "longenough",
      }),
    );
    await waitFor(() => expect(hrefWrites[0]).toBe("/"));
  });

  it("surfaces the server error on failure and does not navigate", async () => {
    vi.mocked(accountsApi.register).mockResolvedValue({
      ok: false,
      error: "invite expired",
      status: 400,
    });
    renderRegisterAt("?invite=tok123");
    fillForm("alice", "longenough", "longenough");
    fireEvent.click(screen.getByRole("button", { name: /create account/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent("invite expired");
    expect(hrefWrites).toHaveLength(0);
  });
});

describe("RegisterPage v2 (?login-v2=1) flow", () => {
  it("shows the 'Join your team' landing first, not the form", () => {
    renderRegisterAt("?invite=tok123&login-v2=1");
    expect(screen.getByRole("heading", { name: /join your team/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /accept invitation/i })).toBeInTheDocument();
    // The form isn't rendered until the invite is accepted.
    expect(screen.queryByLabelText(/username/i)).not.toBeInTheDocument();
  });

  it("advances to the form on Accept, then redeems and navigates home", async () => {
    renderRegisterAt("?invite=tok123&login-v2=1");
    fireEvent.click(screen.getByRole("button", { name: /accept invitation/i }));

    // Form now shown; complete it.
    fillForm("alice", "longenough", "longenough");
    fireEvent.click(screen.getByRole("button", { name: /create account/i }));

    await waitFor(() =>
      expect(accountsApi.register).toHaveBeenCalledWith({
        invite: "tok123",
        username: "alice",
        password: "longenough",
      }),
    );
    await waitFor(() => expect(hrefWrites[0]).toBe("/"));
  });

  it("surfaces a redeem error after Accept and does not navigate", async () => {
    vi.mocked(accountsApi.register).mockResolvedValue({
      ok: false,
      error: "invite expired",
      status: 400,
    });
    renderRegisterAt("?invite=tok123&login-v2=1");
    fireEvent.click(screen.getByRole("button", { name: /accept invitation/i }));
    fillForm("alice", "longenough", "longenough");
    fireEvent.click(screen.getByRole("button", { name: /create account/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent("invite expired");
    expect(hrefWrites).toHaveLength(0);
  });

  it("skips the landing and shows the invite-required error when the token is missing", () => {
    renderRegisterAt("?login-v2=1");
    expect(screen.queryByRole("button", { name: /accept invitation/i })).not.toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent(/invite token/i);
  });
});

describe("RegisterPage with a team invitation", () => {
  it("names the role and the address the invitation is for", async () => {
    vi.mocked(teamApi.previewInvitation).mockResolvedValue({
      team_name: "Acme",
      email: "erin@example.test",
      role: "admin",
      links_account: false,
      expires_at: 9_999_999_999,
    });
    renderRegisterAt("?invite=tok");
    expect(
      await screen.findByText("You were invited to join Acme as an admin, with erin@example.test."),
    ).toBeTruthy();
    expect(teamApi.previewInvitation).toHaveBeenCalledWith("tok");
  });

  it("says where a new member adds a computer, when the server creates computers", async () => {
    const preview = {
      team_name: "Acme",
      email: "erin@example.test",
      role: "member" as const,
      links_account: false,
      expires_at: 9_999_999_999,
    };
    vi.mocked(teamApi.previewInvitation).mockResolvedValue(preview);
    render(
      <CapabilitiesProvider
        info={{ ...FALLBACK_SERVER_INFO, persistent_host_providers: ["cloudflare"] }}
      >
        <MemoryRouter initialEntries={["/register?invite=tok"]}>
          <RegisterPage />
        </MemoryRouter>
      </CapabilitiesProvider>,
    );
    expect((await screen.findByTestId("register-next-computer")).textContent).toBe(
      "Once you have joined, add your own computer in Settings → Computers.",
    );
    cleanup();
    // No computers on this server: nothing to point at.
    renderRegisterAt("?invite=tok");
    await screen.findByText("You were invited to join Acme as a member, with erin@example.test.");
    expect(screen.queryByTestId("register-next-computer")).toBeNull();
  });

  it("shows the server's refusal for another address", async () => {
    vi.mocked(accountsApi.register).mockResolvedValue({
      ok: false,
      error: "this invitation is for another email address; sign in with the invited address",
      status: 403,
    });
    renderRegisterAt("?invite=tok");
    fillForm("dave", "dave-pw-1234", "dave-pw-1234");
    fireEvent.click(screen.getByRole("button", { name: /create account/i }));
    expect((await screen.findByRole("alert")).textContent).toContain("another email address");
    expect(hrefWrites).toEqual([]);
  });
});

describe("RegisterPage with an invitation for an existing account", () => {
  const linkPreview = {
    team_name: "Acme",
    email: "frank@example.test",
    role: "member" as const,
    links_account: true,
    user_id: "frank",
    expires_at: 9_999_999_999,
  };

  beforeEach(() => {
    vi.mocked(teamApi.previewInvitation).mockResolvedValue(linkPreview);
  });

  async function signInForm() {
    renderRegisterAt("?invite=tok");
    expect(await screen.findByRole("heading", { name: "Sign in to join" })).toBeTruthy();
    const username = screen.getByLabelText(/username/i) as HTMLInputElement;
    expect(username.value).toBe("frank");
    expect(username.readOnly).toBe(true);
    // The account exists: no new password to confirm.
    expect(screen.queryByLabelText(/confirm password/i)).toBeNull();
    return username;
  }

  it("asks for the account's password and shows a wrong one's refusal", async () => {
    vi.mocked(accountsApi.login).mockResolvedValue({
      ok: false,
      error: "Invalid username or password.",
      status: 401,
    });
    await signInForm();
    fireEvent.change(screen.getByLabelText(/^password/i), { target: { value: "wrong" } });
    fireEvent.click(screen.getByRole("button", { name: "Sign in and join" }));

    expect((await screen.findByRole("alert")).textContent).toContain(
      "Invalid username or password",
    );
    expect(accountsApi.login).toHaveBeenCalledWith({
      username: "frank",
      password: "wrong",
      invite: "tok",
    });
    expect(accountsApi.register).not.toHaveBeenCalled();
    expect(hrefWrites).toEqual([]);
  });

  it("signs in with the invite and goes home", async () => {
    vi.mocked(accountsApi.login).mockResolvedValue({
      ok: true,
      user: { id: "frank", is_admin: false },
      token: "t",
      expires_in: 3600,
    });
    await signInForm();
    fireEvent.change(screen.getByLabelText(/^password/i), { target: { value: "frank-pw" } });
    fireEvent.click(screen.getByRole("button", { name: "Sign in and join" }));

    await waitFor(() =>
      expect(accountsApi.login).toHaveBeenCalledWith({
        username: "frank",
        password: "frank-pw",
        invite: "tok",
      }),
    );
    await waitFor(() => expect(hrefWrites[0]).toBe("/"));
  });
});
