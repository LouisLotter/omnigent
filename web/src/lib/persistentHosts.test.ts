import { describe, expect, it } from "vitest";

import type { Host } from "@/hooks/useHosts";
import { computerHost } from "@/test/hostFixtures";
import {
  SIGN_IN_NEEDED_MESSAGE,
  hostDisplayLabel,
  hostStatusWord,
  isHostAvailable,
  isPersistentHostChanging,
  persistentHostRefusal,
  persistentHostStateText,
} from "./persistentHosts";

const laptop = (status: Host["status"]): Host => ({
  host_id: "host_laptop",
  name: "laptop",
  owner: "me",
  status,
});

const sandboxHost: Host = {
  host_id: "host_sandbox",
  name: "sandbox-abc",
  owner: "me",
  status: "offline",
  sandbox_provider: "modal",
};

describe("isHostAvailable", () => {
  it.each([
    ["an online laptop", laptop("online"), true],
    ["an offline laptop", laptop("offline"), false],
    ["an offline per-session sandbox", sandboxHost, false],
    ["an asleep computer", computerHost({ state: "asleep" }), true],
    ["a waking computer", computerHost({ state: "waking" }), true],
    ["a computer going to sleep", computerHost({ state: "sleeping" }), true],
    ["an awake computer", computerHost({ state: "awake" }), true],
    ["a failed computer", computerHost({ state: "failed" }), false],
  ])("treats %s as available: %s", (_label, host, available) => {
    expect(isHostAvailable(host)).toBe(available);
  });
});

describe("persistentHostStateText", () => {
  it.each([
    [{ state: "asleep" as const }, "Asleep"],
    [{ state: "awake" as const }, "Awake"],
    [{ state: "sleeping" as const }, "Going to sleep"],
    [{ state: "waking" as const, wake_stage: null }, "Waking…"],
    [
      { state: "waking" as const, wake_stage: "provisioning" as const },
      "Waking… Starting the computer",
    ],
    [{ state: "waking" as const, wake_stage: "starting" as const }, "Waking… Starting Omnigent"],
    [{ state: "waking" as const, wake_stage: "connecting" as const }, "Waking… Connecting"],
    [{ state: "failed" as const, failure: "Quota exceeded." }, "Failed: Quota exceeded."],
    [{ state: "failed" as const, failure: null }, "Failed"],
  ])("describes %o as %s", (status, text) => {
    expect(persistentHostStateText(computerHost(status))).toBe(text);
  });

  it("has no state text for other hosts", () => {
    expect(persistentHostStateText(laptop("online"))).toBeNull();
    expect(persistentHostStateText(sandboxHost)).toBeNull();
  });
});

describe("hostStatusWord", () => {
  it("uses a computer's state while it is not awake and the online status otherwise", () => {
    expect(hostStatusWord(computerHost({ state: "asleep" }))).toBe("asleep");
    expect(hostStatusWord(computerHost({ state: "waking" }))).toBe("waking…");
    expect(hostStatusWord(computerHost({ state: "awake" }))).toBe("online");
    expect(hostStatusWord(laptop("offline"))).toBe("offline");
  });
});

describe("isPersistentHostChanging", () => {
  it("is true only while a computer wakes or goes to sleep", () => {
    expect(isPersistentHostChanging(computerHost({ state: "waking" }))).toBe(true);
    expect(isPersistentHostChanging(computerHost({ state: "sleeping" }))).toBe(true);
    expect(isPersistentHostChanging(computerHost({ state: "asleep" }))).toBe(false);
    expect(isPersistentHostChanging(computerHost({ state: "failed" }))).toBe(false);
    expect(isPersistentHostChanging(laptop("online"))).toBe(false);
  });
});

describe("persistentHostRefusal", () => {
  it("is null while the computer admits work or has not reported", () => {
    expect(persistentHostRefusal(computerHost())).toBeNull();
    expect(
      persistentHostRefusal(
        computerHost({ admission: { admitting: true, reason: null, message: null } }),
      ),
    ).toBeNull();
  });

  it("asks for a sign-in when the provider needs one", () => {
    const host = computerHost({
      admission: { admitting: false, reason: "needs_sign_in", message: "Codex: signed out" },
    });
    expect(persistentHostRefusal(host)).toEqual({
      message: SIGN_IN_NEEDED_MESSAGE,
      needsSignIn: true,
    });
  });

  it("shows the provider's message for other refusals, with a fallback", () => {
    const withMessage = computerHost({
      admission: { admitting: false, reason: "not_saved", message: "Files not saved for 2 h." },
    });
    expect(persistentHostRefusal(withMessage)).toEqual({
      message: "Files not saved for 2 h.",
      needsSignIn: false,
    });
    const withoutMessage = computerHost({
      admission: { admitting: false, reason: null, message: null },
    });
    expect(persistentHostRefusal(withoutMessage)?.message).toBe(
      "This computer isn't accepting new work right now.",
    );
  });
});

describe("hostDisplayLabel", () => {
  it("names per-session sandboxes by provider and computers by their own name", () => {
    expect(hostDisplayLabel(sandboxHost)).toBe("Modal Sandbox");
    expect(hostDisplayLabel(computerHost())).toBe("research-box");
    expect(hostDisplayLabel(laptop("online"))).toBe("laptop");
  });
});
