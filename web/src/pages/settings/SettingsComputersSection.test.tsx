import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Host } from "@/hooks/useHosts";
import { CapabilitiesProvider } from "@/lib/CapabilitiesContext";
import { FALLBACK_SERVER_INFO } from "@/lib/capabilities";
import { authenticatedFetch } from "@/lib/identity";
import { SIGN_IN_NEEDED_MESSAGE } from "@/lib/persistentHosts";
import { computerHost } from "@/test/hostFixtures";
import { SettingsComputersSection } from "./SettingsComputersSection";

vi.mock("@/lib/identity", () => ({ authenticatedFetch: vi.fn() }));

const fetchMock = vi.mocked(authenticatedFetch);
const response = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

const MANAGE_URL = "https://computers.example.test/manage/host_computer";
const laptop: Host = { host_id: "host_laptop", name: "laptop", owner: "me", status: "online" };

let hosts: Host[];
let startResponse: () => Response;
let deleteResponse: () => Response;

function requests(method: string, path: string) {
  return fetchMock.mock.calls.filter(
    ([url, init]) => url === path && (init?.method ?? "GET") === method,
  );
}

function renderSection(providers = ["cloudflare"]) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <CapabilitiesProvider
        info={{ ...FALLBACK_SERVER_INFO, persistent_host_providers: providers }}
      >
        <SettingsComputersSection />
      </CapabilitiesProvider>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  hosts = [];
  startResponse = () => response(computerHost({ state: "waking" }), 202);
  deleteResponse = () => response({ host_id: "host_computer", deleted: true });
  fetchMock.mockReset();
  fetchMock.mockImplementation(async (input, init) => {
    const method = init?.method ?? "GET";
    if (input === "/v1/hosts" && method === "GET") return response({ hosts });
    if (input === "/v1/hosts/host_computer/start" && method === "POST") return startResponse();
    if (input === "/v1/hosts/host_computer" && method === "DELETE") return deleteResponse();
    throw new Error(`unexpected ${method} ${String(input)}`);
  });
});
afterEach(cleanup);

describe("SettingsComputersSection", () => {
  it("lists only computers, each with its state and wake progress", async () => {
    hosts = [
      laptop,
      computerHost({ state: "asleep" }, { host_id: "host_asleep", name: "asleep-box" }),
      computerHost(
        { state: "waking", wake_stage: "provisioning" },
        { host_id: "host_waking", name: "waking-box" },
      ),
      computerHost({ state: "sleeping" }, { host_id: "host_sleeping", name: "sleeping-box" }),
    ];
    renderSection();

    const state = async (hostId: string) =>
      within(await screen.findByTestId(`computer-${hostId}`)).getByTestId("computer-state");
    expect(await state("host_asleep")).toHaveTextContent("Asleep");
    expect(await state("host_waking")).toHaveTextContent("Waking…");
    expect(await state("host_waking")).toHaveTextContent("Starting the computer");
    expect(await state("host_sleeping")).toHaveTextContent("Going to sleep");
    expect(screen.queryByTestId("computer-host_laptop")).toBeNull();
    expect(screen.queryByText("laptop")).toBeNull();
  });

  it("starts a failed computer again and shows its new state", async () => {
    hosts = [computerHost({ state: "failed", failure: "The computer no longer exists." })];
    renderSection();

    const row = await screen.findByTestId("computer-host_computer");
    expect(within(row).getByTestId("computer-state")).toHaveTextContent(
      "The computer no longer exists.",
    );
    hosts = [computerHost({ state: "waking", wake_stage: "starting" })];
    fireEvent.click(within(row).getByRole("button", { name: "Start again" }));

    await waitFor(() =>
      expect(within(row).getByTestId("computer-state")).toHaveTextContent("Starting Omnigent"),
    );
    expect(requests("POST", "/v1/hosts/host_computer/start")).toHaveLength(1);
    expect(within(row).queryByRole("button", { name: "Start again" })).toBeNull();
  });

  it("shows why a start again was refused", async () => {
    hosts = [computerHost({ state: "failed", failure: "Setup was interrupted." })];
    startResponse = () => response({ detail: "The provider is unavailable." }, 400);
    renderSection();

    const row = await screen.findByTestId("computer-host_computer");
    fireEvent.click(within(row).getByRole("button", { name: "Start again" }));

    expect(await within(row).findByRole("alert")).toHaveTextContent("The provider is unavailable.");
    expect(within(row).getByRole("button", { name: "Start again" })).toBeEnabled();
  });

  it("asks for a sign-in and links the computer's setup page", async () => {
    hosts = [
      computerHost({
        state: "awake",
        manage_url: MANAGE_URL,
        admission: { admitting: false, reason: "needs_sign_in", message: null },
      }),
    ];
    renderSection();

    const row = await screen.findByTestId("computer-host_computer");
    expect(within(row).getByTestId("computer-refusal")).toHaveTextContent(SIGN_IN_NEEDED_MESSAGE);
    const link = within(row).getByRole("link", { name: "Set up / manage" });
    expect(link).toHaveAttribute("href", MANAGE_URL);
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("shows the provider's message for another refusal", async () => {
    hosts = [
      computerHost({
        admission: { admitting: false, reason: "not_saved", message: "Files not saved recently." },
      }),
    ];
    renderSection();

    const row = await screen.findByTestId("computer-host_computer");
    expect(within(row).getByTestId("computer-refusal")).toHaveTextContent(
      "Files not saved recently.",
    );
    expect(within(row).queryByRole("link")).toBeNull();
  });

  it("points to signing out before deleting, then deletes the computer", async () => {
    hosts = [computerHost({ manage_url: MANAGE_URL })];
    renderSection();

    const row = await screen.findByTestId("computer-host_computer");
    fireEvent.click(within(row).getByRole("button", { name: "Delete" }));
    const dialog = screen.getByTestId("delete-computer-dialog");
    expect(dialog).toHaveTextContent("Deleting removes the computer and its files.");
    expect(within(dialog).getByTestId("delete-computer-sign-out")).toHaveTextContent(
      "Before deleting, sign out of the accounts you use on this computer: open Set up / manage.",
    );
    expect(within(dialog).getByRole("link", { name: "Set up / manage" })).toHaveAttribute(
      "href",
      MANAGE_URL,
    );

    hosts = [];
    fireEvent.click(within(dialog).getByTestId("delete-computer-confirm"));

    await waitFor(() => expect(screen.queryByTestId("delete-computer-dialog")).toBeNull());
    expect(requests("DELETE", "/v1/hosts/host_computer")).toHaveLength(1);
    expect(screen.queryByTestId("computer-host_computer")).toBeNull();
    expect(await screen.findByTestId("computers-empty")).toBeInTheDocument();
  });

  it("keeps the delete dialog open with the server's reason when deleting fails", async () => {
    hosts = [computerHost()];
    deleteResponse = () =>
      response({ error: { code: "forbidden", message: "not your host" } }, 403);
    renderSection();

    fireEvent.click(
      within(await screen.findByTestId("computer-host_computer")).getByRole("button", {
        name: "Delete",
      }),
    );
    const dialog = screen.getByTestId("delete-computer-dialog");
    expect(within(dialog).getByTestId("delete-computer-sign-out")).toHaveTextContent(
      "Before deleting, sign out of the accounts you use on this computer.",
    );
    expect(within(dialog).queryByRole("link")).toBeNull();
    fireEvent.click(within(dialog).getByTestId("delete-computer-confirm"));

    expect(await within(dialog).findByTestId("delete-computer-error")).toHaveTextContent(
      "not your host",
    );
    expect(screen.getByTestId("computer-host_computer")).toBeInTheDocument();
  });

  it("offers Add computer only when a provider creates computers", async () => {
    renderSection([]);
    expect(await screen.findByTestId("computers-empty")).toBeInTheDocument();
    expect(screen.queryByTestId("add-computer")).toBeNull();
    cleanup();

    renderSection(["cloudflare"]);
    fireEvent.click(screen.getByTestId("add-computer"));
    expect(screen.getByTestId("add-computer-dialog")).toBeInTheDocument();
  });
});
