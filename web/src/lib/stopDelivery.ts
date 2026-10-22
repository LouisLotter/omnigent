import type { PostEventResponse } from "./sessionsApi";

/** Shown when a Stop did not reach the computer; the server never sends it again. */
export const STOP_NOT_DELIVERED_MESSAGE =
  "Stop didn't reach the computer. If the turn is still running, press Stop again.";

/**
 * Whether an interrupt's answer says the Stop may not have reached the computer.
 *
 * @param delivery - The interrupt answer's `delivery`, e.g. `"unknown"`.
 * @returns `true` for `"not_delivered"` and `"unknown"`.
 */
export function stopDidNotReachComputer(delivery: PostEventResponse["delivery"]): boolean {
  return delivery === "not_delivered" || delivery === "unknown";
}
