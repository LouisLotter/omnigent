import { ExternalLinkIcon, TriangleAlertIcon } from "lucide-react";

import type { Host } from "@/hooks/useHosts";
import {
  isPersistentHostFailed,
  persistentHostManageUrl,
  persistentHostRefusal,
} from "@/lib/persistentHosts";
import { cn } from "@/lib/utils";

/** Opens a computer's provider setup page in a new tab. */
export function ManageComputerLink({ url, className }: { url: string; className?: string }) {
  return (
    <a
      href={url}
      target="_blank"
      rel="noopener noreferrer"
      className={cn(
        "inline-flex items-center gap-1 underline underline-offset-2 hover:no-underline",
        className,
      )}
    >
      Set up / manage
      <ExternalLinkIcon aria-hidden className="size-3 shrink-0" />
    </a>
  );
}

/**
 * Why a selected computer cannot take work: it failed to start (with a
 * "Start again" action) or it refuses new work (with its setup link).
 * Renders nothing for any other host.
 */
export function ComputerAttentionNotice({
  host,
  onStartAgain,
  starting,
  startError,
}: {
  host: Host;
  onStartAgain: () => void;
  starting: boolean;
  startError: string | null;
}) {
  if (isPersistentHostFailed(host)) {
    const failure = host.persistent_status?.failure;
    return (
      <p
        role="alert"
        className="flex flex-wrap items-center gap-x-2 pl-2 text-xs text-destructive"
        data-testid="computer-attention"
      >
        <TriangleAlertIcon aria-hidden className="size-3.5 shrink-0" />
        <span>
          {failure ? `${host.name} failed to start: ${failure}` : `${host.name} failed to start.`}
        </span>
        <button
          type="button"
          className="underline underline-offset-2 hover:no-underline disabled:opacity-60"
          disabled={starting}
          onClick={onStartAgain}
        >
          Start again
        </button>
        {startError && <span>{startError}</span>}
      </p>
    );
  }
  const refusal = persistentHostRefusal(host);
  if (!refusal) return null;
  const manageUrl = persistentHostManageUrl(host);
  return (
    <p
      className="flex flex-wrap items-center gap-x-2 pl-2 text-xs text-amber-600 dark:text-amber-500"
      data-testid="computer-attention"
    >
      <TriangleAlertIcon aria-hidden className="size-3.5 shrink-0" />
      <span>{refusal.message}</span>
      {manageUrl && <ManageComputerLink url={manageUrl} />}
    </p>
  );
}
