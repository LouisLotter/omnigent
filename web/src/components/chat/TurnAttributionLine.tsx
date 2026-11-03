// One plain line saying who started a turn, on whose computer and harness
// login: "Bob, on Alice's computer and Claude subscription". The full record
// (thread owner, computer, login state, harness, model, effort) is in the
// tooltip. Shown in the transcript when someone other than the computer's
// owner, a schedule or an agent started the turn; otherwise in the bubble's
// hover row.

import { UserRoundCogIcon } from "lucide-react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { summarizeAttribution, type TurnAttribution } from "@/lib/attribution";
import { cn } from "@/lib/utils";

export function TurnAttributionLine({
  attribution,
  viewer,
  placement,
}: {
  attribution: TurnAttribution;
  viewer: string | null;
  /** `"transcript"` renders only prominent lines; `"hover"` only the others. */
  placement: "transcript" | "hover";
}) {
  const summary = summarizeAttribution(attribution, viewer);
  if (summary.prominent !== (placement === "transcript")) return null;
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          tabIndex={0}
          data-testid="turn-attribution"
          data-placement={placement}
          className={cn(
            "inline-flex min-w-0 items-center gap-1 text-[11px] leading-4 text-muted-foreground",
            placement === "transcript" && "pt-1",
          )}
        >
          <UserRoundCogIcon size={12} aria-hidden className="shrink-0" />
          <span className="truncate">{summary.text}</span>
        </span>
      </TooltipTrigger>
      <TooltipContent className="max-w-96">
        <ul className="flex flex-col gap-0.5" data-testid="turn-attribution-detail">
          {summary.detail.map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ul>
      </TooltipContent>
    </Tooltip>
  );
}
