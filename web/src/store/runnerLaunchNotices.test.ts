import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import {
  clearRunnerLaunchNotice,
  noteRunnerLaunch,
  useRunnerLaunchNotice,
} from "./runnerLaunchNotices";

describe("runner launch notices", () => {
  afterEach(() => {
    clearRunnerLaunchNotice("conv_a");
    clearRunnerLaunchNotice("conv_b");
  });

  it.each([
    { outcome: "failed", notice: "failed" },
    { outcome: "unknown", notice: "unknown" },
    { outcome: "started", notice: null },
    { outcome: null, notice: null },
    { outcome: undefined, notice: null },
  ])("records a create's runner_launch $outcome as $notice", ({ outcome, notice }) => {
    const { result } = renderHook(() => useRunnerLaunchNotice("conv_a"));

    act(() => noteRunnerLaunch("conv_a", outcome));

    expect(result.current).toBe(notice);
  });

  it("keeps each session's notice apart and clears one without the other", () => {
    const { result } = renderHook(() => ({
      a: useRunnerLaunchNotice("conv_a"),
      b: useRunnerLaunchNotice("conv_b"),
      none: useRunnerLaunchNotice(null),
    }));

    act(() => {
      noteRunnerLaunch("conv_a", "failed");
      noteRunnerLaunch("conv_b", "unknown");
    });
    expect(result.current).toEqual({ a: "failed", b: "unknown", none: null });

    act(() => clearRunnerLaunchNotice("conv_a"));
    expect(result.current).toEqual({ a: null, b: "unknown", none: null });
  });
});
