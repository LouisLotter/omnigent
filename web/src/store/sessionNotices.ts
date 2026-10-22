import { create } from "zustand";

/** One notice per session, kept apart from the session's stream state. */
export interface SessionNotices<Notice extends string> {
  /** Show *notice* for *sessionId*, replacing any earlier one. */
  set: (sessionId: string, notice: Notice) => void;
  /** Forget *sessionId*'s notice. */
  clear: (sessionId: string) => void;
  /** *sessionId*'s notice, or `null`. */
  get: (sessionId: string) => Notice | null;
  /** React hook: *sessionId*'s notice, or `null` (always `null` for a `null` id). */
  useNotice: (sessionId: string | null) => Notice | null;
}

/**
 * Create a store of per-session notices.
 *
 * A conversation's stream entry can be evicted while it is in the background;
 * a notice the person has not seen yet must outlive that, so it lives here.
 */
export function createSessionNotices<Notice extends string>(): SessionNotices<Notice> {
  const useStore = create<{ bySession: Record<string, Notice> }>(() => ({ bySession: {} }));
  return {
    set: (sessionId, notice) =>
      useStore.setState((state) =>
        state.bySession[sessionId] === notice
          ? state
          : { bySession: { ...state.bySession, [sessionId]: notice } },
      ),
    clear: (sessionId) =>
      useStore.setState((state) => {
        if (!(sessionId in state.bySession)) return state;
        const { [sessionId]: _cleared, ...rest } = state.bySession;
        return { bySession: rest };
      }),
    get: (sessionId) => useStore.getState().bySession[sessionId] ?? null,
    useNotice: (sessionId) =>
      useStore((state) => (sessionId === null ? null : (state.bySession[sessionId] ?? null))),
  };
}
