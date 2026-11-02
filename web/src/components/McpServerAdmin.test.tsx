import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { McpServerAdmin, serviceIdFrom, settingsFrom } from "./McpServerAdmin";
import { getCurrentIsAdmin, resolveIdentity } from "@/lib/identity";
import {
  AdminRequestError,
  adminRequest,
  authorizeRegistryService,
  listAdminServices,
  registryRequest,
  type AdminService,
} from "@/lib/mcpRegistry";
import type * as McpRegistryModule from "@/lib/mcpRegistry";

vi.mock("@/lib/identity", () => ({ getCurrentIsAdmin: vi.fn(), resolveIdentity: vi.fn() }));
vi.mock("@/lib/mcpRegistry", async (importOriginal) => ({
  ...(await importOriginal<typeof McpRegistryModule>()),
  adminRequest: vi.fn(),
  listAdminServices: vi.fn(),
  registryRequest: vi.fn(),
  authorizeRegistryService: vi.fn(),
}));

const tracker: AdminService = {
  id: "tracker",
  title: "Work tracker",
  description: "Tickets",
  url: "https://tracker.example.com/mcp",
  auth: "bearer",
  timeout: 60,
  tools: ["read_ticket"],
  source: "admin",
  editable: true,
  oauth: null,
  draft: false,
};

const shipped: AdminService = {
  ...tracker,
  id: "github",
  title: "GitHub",
  auth: "github",
  source: "config",
  editable: false,
};

function presetsAnd(...responses: unknown[]) {
  const mock = vi.mocked(adminRequest);
  mock.mockImplementation(async (path?: string) => {
    if (path === "/presets") return { data: [] };
    const next = responses.shift();
    if (next instanceof Error) throw next;
    return next;
  });
  return mock;
}

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(getCurrentIsAdmin).mockReturnValue(true);
  vi.mocked(resolveIdentity).mockResolvedValue("admin");
});

describe("serviceIdFrom", () => {
  it.each([
    ["Work Tracker", "work-tracker"],
    ["  GitHub (org) ", "github-org"],
    ["123 Tools", "mcp-123-tools"],
    ["***", ""],
    ["a".repeat(60), "a".repeat(40)],
  ])("%s becomes %s", (name, id) => {
    expect(serviceIdFrom(name)).toBe(id);
  });
});

describe("settingsFrom", () => {
  it("keeps a registered client's endpoints without sending its client ID", () => {
    const service: AdminService = {
      ...tracker,
      auth: "oauth",
      oauth: {
        authorize_url: "https://auth.example.com/authorize",
        token_url: "https://auth.example.com/token",
        client_id: "dyn-1",
        scopes: ["read"],
        resource: null,
        has_client_secret: true,
        registration: "dynamic",
      },
    };
    expect(settingsFrom(service, { tools: ["a"] })).toEqual({
      title: "Work tracker",
      description: "Tickets",
      url: "https://tracker.example.com/mcp",
      auth: "oauth",
      timeout: 60,
      tools: ["a"],
      oauth: {
        authorize_url: "https://auth.example.com/authorize",
        token_url: "https://auth.example.com/token",
        scopes: ["read"],
      },
    });
  });
});

describe("MCP server administration", () => {
  it("shows nothing to members or on servers without run-time services", async () => {
    vi.mocked(getCurrentIsAdmin).mockReturnValue(false);
    const member = render(<McpServerAdmin />);
    await waitFor(() => expect(resolveIdentity).toHaveBeenCalled());
    expect(member.container).toBeEmptyDOMElement();
    expect(listAdminServices).not.toHaveBeenCalled();
    member.unmount();

    vi.mocked(getCurrentIsAdmin).mockReturnValue(true);
    vi.mocked(listAdminServices).mockRejectedValue(new AdminRequestError("Not Found", 404));
    presetsAnd();
    const { container } = render(<McpServerAdmin />);
    await waitFor(() => expect(listAdminServices).toHaveBeenCalled());
    expect(container).toBeEmptyDOMElement();
  });

  it("lists page-added servers as editable and configured ones as read-only", async () => {
    vi.mocked(listAdminServices).mockResolvedValue([shipped, tracker]);
    presetsAnd();
    render(<McpServerAdmin />);
    expect(await screen.findByText(/from the server's configuration/)).toBeInTheDocument();
    expect(screen.getByText(/1 tool allowed/)).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: "Edit" })).toHaveLength(1);
  });

  it("adds a server, saves the admin's token, discovers tools and saves the allowlist", async () => {
    const draft: AdminService = { ...tracker, tools: [], draft: true };
    vi.mocked(listAdminServices).mockResolvedValue([]);
    const requests = presetsAnd(
      draft,
      {
        tools: [
          { name: "read_ticket", description: "Read one" },
          { name: "delete_ticket", description: "Delete one" },
        ],
        allowed: [],
      },
      { ...tracker },
    );
    vi.mocked(registryRequest).mockResolvedValue({ connected: true });
    const onChanged = vi.fn();
    render(<McpServerAdmin onChanged={onChanged} />);

    fireEvent.click(await screen.findByRole("button", { name: "Add MCP server" }));
    const dialog = await screen.findByRole("dialog");
    fireEvent.change(within(dialog).getByLabelText("Name"), { target: { value: "Work tracker" } });
    expect(within(dialog).getByLabelText("ID")).toHaveValue("work-tracker");
    fireEvent.change(within(dialog).getByLabelText("ID"), { target: { value: "tracker" } });
    fireEvent.change(within(dialog).getByLabelText("Server URL"), {
      target: { value: "https://tracker.example.com/mcp" },
    });
    fireEvent.click(within(dialog).getByRole("radio", { name: "API token" }));
    fireEvent.click(within(dialog).getByRole("button", { name: "Save and continue" }));

    await screen.findByRole("button", { name: "Discover tools" });
    expect(requests).toHaveBeenCalledWith("/services", expect.objectContaining({ method: "POST" }));
    const created = JSON.parse(
      requests.mock.calls.find(([path]) => path === "/services")![1]!.body as string,
    );
    expect(created).toMatchObject({
      id: "tracker",
      title: "Work tracker",
      url: "https://tracker.example.com/mcp",
      auth: "bearer",
      tools: [],
      oauth: null,
    });

    const token = screen.getByLabelText("Your token for Work tracker");
    expect(token).toHaveAttribute("type", "password");
    fireEvent.change(token, { target: { value: "secret-token" } });
    fireEvent.click(screen.getByRole("button", { name: "Save my token" }));
    await screen.findByText("Your token is saved as your own login.");
    expect(registryRequest).toHaveBeenCalledWith(
      "/tracker/connection",
      expect.objectContaining({ method: "PUT", body: JSON.stringify({ token: "secret-token" }) }),
    );
    expect(token).toHaveValue("");

    fireEvent.click(screen.getByRole("button", { name: "Discover tools" }));
    const remove = await screen.findByRole("checkbox", { name: "delete_ticket" });
    expect(remove).toBeChecked();
    fireEvent.click(remove);
    fireEvent.click(screen.getByRole("button", { name: "Save allowed tools" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    const update = requests.mock.calls.find(([, init]) => init?.method === "PUT")!;
    expect(update[0]).toBe("/services/tracker");
    expect(JSON.parse(update[1]!.body as string).tools).toEqual(["read_ticket"]);
    expect(onChanged).toHaveBeenCalled();
  });

  it("signs the admin in through the shared popup flow for an OAuth server", async () => {
    const oauthDraft: AdminService = {
      ...tracker,
      auth: "oauth",
      tools: [],
      draft: true,
      oauth: {
        authorize_url: "https://auth.example.com/authorize",
        token_url: "https://auth.example.com/token",
        client_id: "dyn-1",
        scopes: [],
        resource: null,
        has_client_secret: true,
        registration: "dynamic",
      },
    };
    vi.mocked(listAdminServices).mockResolvedValue([oauthDraft]);
    presetsAnd();
    vi.mocked(authorizeRegistryService).mockResolvedValue();
    render(<McpServerAdmin />);
    fireEvent.click(await screen.findByRole("button", { name: "Finish setup" }));
    fireEvent.click(await screen.findByRole("button", { name: "Sign in with your account" }));
    await screen.findByText("Signed in with your account.");
    expect(vi.mocked(authorizeRegistryService).mock.calls[0][0]).toMatchObject({
      id: "tracker",
      connect_provider: "mcp-tracker",
    });
  });

  it("warns that a new URL removes saved logins, and confirms before removing", async () => {
    vi.mocked(listAdminServices).mockResolvedValue([tracker]);
    const requests = presetsAnd({ deleted: true });
    render(<McpServerAdmin />);
    fireEvent.click(await screen.findByRole("button", { name: "Edit" }));
    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).queryByRole("note")).toBeNull();
    fireEvent.change(within(dialog).getByLabelText("Server URL"), {
      target: { value: "https://elsewhere.example.com/mcp" },
    });
    expect(within(dialog).getByRole("note")).toHaveTextContent(
      "removes every member's saved login",
    );
    fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }));

    fireEvent.click(screen.getByRole("button", { name: "Remove" }));
    expect(requests).not.toHaveBeenCalledWith("/services/tracker", expect.anything());
    fireEvent.click(screen.getByRole("button", { name: "Remove Work tracker" }));
    await waitFor(() =>
      expect(requests).toHaveBeenCalledWith(
        "/services/tracker",
        expect.objectContaining({ method: "DELETE" }),
      ),
    );
  });

  it("shows the server's explanation when a save is refused", async () => {
    vi.mocked(listAdminServices).mockResolvedValue([]);
    presetsAnd(
      new AdminRequestError("The server URL breaks a destination rule: it must be https.", 400),
    );
    render(<McpServerAdmin />);
    fireEvent.click(await screen.findByRole("button", { name: "Add MCP server" }));
    const dialog = await screen.findByRole("dialog");
    fireEvent.change(within(dialog).getByLabelText("Name"), { target: { value: "Plain" } });
    fireEvent.change(within(dialog).getByLabelText("Server URL"), {
      target: { value: "http://plain.example.com/mcp" },
    });
    fireEvent.click(within(dialog).getByRole("button", { name: "Save and continue" }));
    expect(await within(dialog).findByRole("alert")).toHaveTextContent("it must be https");
  });
});
