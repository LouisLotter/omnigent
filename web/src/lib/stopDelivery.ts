import type { PostEventResponse } from "./sessionsApi";

/**
 * Why the last Stop may not have stopped the turn. A Stop is never sent again,
 * so each notice tells the person to press Interrupt again.
 *
 * - `not_delivered`: no runner connection took it, or its answer was lost.
 * - `refused`: the runner answered with an error.
 * - `superseded`: a newer message started a turn first, so it was not sent.
 * - `request_failed`: the request itself failed (network, 502), so whether it
 *   reached the computer is unknown.
 */
export type StopNotice = "not_delivered" | "refused" | "superseded" | "request_failed";

const PRESS_AGAIN = "If the turn is still running, press Interrupt again.";

/** The text shown for each {@link StopNotice}. */
export const STOP_NOTICE_MESSAGES: Record<StopNotice, string> = {
  not_delivered: `The interrupt didn't reach the computer. ${PRESS_AGAIN}`,
  refused: `The computer refused the interrupt. ${PRESS_AGAIN}`,
  superseded: `A newer message arrived first, so the interrupt wasn't sent. ${PRESS_AGAIN}`,
  request_failed: `The interrupt may not have reached the computer. ${PRESS_AGAIN}`,
};

/**
 * The notice an interrupt's answer calls for.
 *
 * @param delivery - The interrupt answer's `delivery`, e.g. `"unknown"`.
 * @returns The notice, or `null` when the Stop was delivered or the server
 *   reported no outcome.
 */
export function stopNoticeForDelivery(delivery: PostEventResponse["delivery"]): StopNotice | null {
  switch (delivery) {
    case "not_delivered":
    case "unknown":
      return "not_delivered";
    case "failed":
      return "refused";
    case "superseded":
      return "superseded";
    default:
      return null;
  }
}
