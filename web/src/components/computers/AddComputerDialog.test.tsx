import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { authenticatedFetch } from "@/lib/identity";
import { computerHost } from "@/test/hostFixtures";
import { AddComputerDialog } from "./AddComputerDialog";

vi.mock("@/lib/identity", () => ({ authenticatedFetch: vi.fn() }));

const fetchMock = vi.mocked(authenticatedFetch);
const response = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

function renderDialog(providers: string[]) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const invalidate = vi.spyOn(client, "invalidateQueries");
  const onOpenChange = vi.fn();
  const onCreated = vi.fn();
  render(
    <QueryClientProvider client={client}>
      <AddComputerDialog
        open
        onOpenChange={onOpenChange}
        providers={providers}
        onCreated={onCreated}
      />
    </QueryClientProvider>,
  );
  return { invalidate, onOpenChange, onCreated };
}

function typeName(name: string) {
  fireEvent.change(screen.getByTestId("add-computer-name"), { target: { value: name } });
}

function postedBody(): unknown {
  const [path, init] = fetchMock.mock.calls[0];
  expect(path).toBe("/v1/hosts");
  expect(init?.method).toBe("POST");
  return JSON.parse(init?.body as string);
}

beforeEach(() => fetchMock.mockReset());
afterEach(cleanup);

describe("AddComputerDialog", () => {
  it("creates the computer on the only provider, then closes and refreshes the hosts", async () => {
    const created = computerHost({ state: "waking" }, { name: "Research box" });
    fetchMock.mockResolvedValue(response(created, 201));
    const { invalidate, onOpenChange, onCreated } = renderDialog(["cloudflare"]);

    expect(screen.queryByTestId("add-computer-provider")).toBeNull();
    typeName("  Research box  ");
    fireEvent.click(screen.getByTestId("add-computer-submit"));

    await waitFor(() => expect(onCreated).toHaveBeenCalledWith(created));
    expect(postedBody()).toEqual({ name: "Research box", provider: "cloudflare" });
    expect(onOpenChange).toHaveBeenCalledWith(false);
    expect(invalidate).toHaveBeenCalledWith({ queryKey: ["hosts"] });
  });

  it("asks for the provider when several create computers", async () => {
    fetchMock.mockResolvedValue(response(computerHost(), 201));
    const { onCreated } = renderDialog(["cloudflare", "modal"]);

    const trigger = screen.getByTestId("add-computer-provider");
    expect(trigger).toHaveTextContent("Cloudflare");
    fireEvent.pointerDown(trigger, new MouseEvent("pointerdown", { bubbles: true, button: 0 }));
    fireEvent.click(trigger);
    fireEvent.click(screen.getByRole("option", { name: "Modal" }));
    typeName("Build box");
    fireEvent.click(screen.getByTestId("add-computer-submit"));

    await waitFor(() => expect(onCreated).toHaveBeenCalled());
    expect(postedBody()).toEqual({ name: "Build box", provider: "modal" });
  });

  it("requires a name of at most 64 characters", () => {
    renderDialog(["cloudflare"]);
    const submit = screen.getByTestId("add-computer-submit");

    expect(submit).toBeDisabled();
    typeName("   ");
    expect(submit).toBeDisabled();
    typeName("x".repeat(65));
    expect(screen.getByRole("alert")).toHaveTextContent("Use 64 characters or fewer.");
    expect(submit).toBeDisabled();
    fireEvent.submit(submit.closest("form")!);
    expect(fetchMock).not.toHaveBeenCalled();

    typeName("x".repeat(64));
    expect(screen.queryByRole("alert")).toBeNull();
    expect(submit).toBeEnabled();
  });

  it("shows a name conflict inline and stays open", async () => {
    fetchMock.mockResolvedValue(
      response({ detail: "You already have a computer named Research box." }, 409),
    );
    const { onOpenChange, onCreated } = renderDialog(["cloudflare"]);

    typeName("Research box");
    fireEvent.click(screen.getByTestId("add-computer-submit"));

    expect(await screen.findByTestId("add-computer-error")).toHaveTextContent(
      "You already have a computer named Research box.",
    );
    expect(onCreated).not.toHaveBeenCalled();
    expect(onOpenChange).not.toHaveBeenCalled();
    expect(screen.getByTestId("add-computer-submit")).toBeEnabled();
  });
});
