// Who asked for an approval and who answered under which grant, read from the
// server's approval record: "Asked during Bob's turn, on Alice's computer",
// "Approved by Alice as the thread's owner". Rendered under an approval card
// and on the standalone approval page. Prompts that are not recorded
// approvals (questions, plan reviews, harness prompts) show nothing.

import { useEffect, useState } from "react";
import { getCurrentAuthorId, authenticatedFetch } from "@/lib/identity";
import { parseApprovalRecord, summarizeApproval, type ApprovalRecord } from "@/lib/attribution";

/**
 * Read one recorded approval, or `null` when it is not one (a question, a plan
 * review, a harness prompt) or the caller may not read it. The server allows
 * readers of the thread and the approvers named to answer it.
 */
export async function fetchApprovalRecord(
  sessionId: string,
  elicitationId: string,
): Promise<ApprovalRecord | null> {
  const res = await authenticatedFetch(
    `/v1/sessions/${encodeURIComponent(sessionId)}/approvals/${encodeURIComponent(elicitationId)}`,
  );
  if (!res.ok) return null;
  return parseApprovalRecord(await res.json());
}

/** Fetch an approval's record; refetched when its card's status changes. */
export function useApprovalRecord(
  sessionId: string | null | undefined,
  elicitationId: string,
  status: string,
): ApprovalRecord | null {
  const [record, setRecord] = useState<ApprovalRecord | null>(null);
  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;
    void (async () => {
      try {
        const res = await fetchApprovalRecord(sessionId, elicitationId);
        if (!cancelled) setRecord(res);
      } catch {
        // Attribution is informational; a failed read leaves the card as it was.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [sessionId, elicitationId, status]);
  return record;
}

export function ApprovalAttribution({ record }: { record: ApprovalRecord | null }) {
  if (record === null) return null;
  const lines = summarizeApproval(record, getCurrentAuthorId());
  if (lines.length === 0) return null;
  return (
    <ul
      className="flex flex-col gap-0.5 px-1 pt-1 text-[11px] leading-4 text-muted-foreground"
      data-testid="approval-attribution"
      title={`Decision digest ${record.decision_digest}`}
    >
      {lines.map((line) => (
        <li key={line}>{line}</li>
      ))}
    </ul>
  );
}
