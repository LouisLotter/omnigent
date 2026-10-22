import { create } from "zustand";

/** A session create's `runner_launch` that did not confirm the runner started. */
export type RunnerLaunchNotice = "failed" | "unknown";

/** The text shown for each {@link RunnerLaunchNotice}. */
export const RUNNER_LAUNCH_MESSAGES: Record<RunnerLaunchNotice, string> = {
  failed: "The agent didn't start on the computer. Sending a message starts it again.",
  unknown:
    "The computer didn't confirm that the agent started. Sending a message starts it if it isn't running.",
};

interface RunnerLaunchNoticeState {
  /** Session id -> the launch outcome its create reported. */
  bySession: Record<string, RunnerLaunchNotice>;
}

const useRunnerLaunchNoticeStore = create<RunnerLaunchNoticeState>(() => ({ bySession: {} }));

/**
 * Record the runner launch a session create reported.
 *
 * The create answers 201 for the saved session whatever happened to its
 * launch, so only `failed` and `unknown` need saying; anything else is ignored.
 *
 * @param sessionId - The created session, e.g. `"conv_abc"`.
 * @param outcome - The create response's `runner_launch`, e.g. `"failed"`.
 */
export function noteRunnerLaunch(sessionId: string, outcome: unknown): void {
  if (outcome !== "failed" && outcome !== "unknown") return;
  useRunnerLaunchNoticeStore.setState((state) => ({
    bySession: { ...state.bySession, [sessionId]: outcome },
  }));
}

/** Forget *sessionId*'s launch notice, e.g. once its runner is online. */
export function clearRunnerLaunchNotice(sessionId: string): void {
  useRunnerLaunchNoticeStore.setState((state) => {
    if (!(sessionId in state.bySession)) return state;
    const { [sessionId]: _cleared, ...rest } = state.bySession;
    return { bySession: rest };
  });
}

/** The launch notice for *sessionId*, or `null`. */
export function useRunnerLaunchNotice(sessionId: string | null): RunnerLaunchNotice | null {
  return useRunnerLaunchNoticeStore((state) =>
    sessionId === null ? null : (state.bySession[sessionId] ?? null),
  );
}
