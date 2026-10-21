import { MonitorCloudIcon, MonitorIcon } from "lucide-react";

import type { Host } from "@/hooks/useHosts";
import { hostStatusWord, isPersistentHost } from "@/lib/persistentHosts";

/**
 * Compact host row for a `<Select>` item — icon, name, and an
 * online/offline pill (a computer's state while it is not awake).
 *
 * Its own module rather than a sibling dialog's export: the dialogs that
 * render host pickers also reference each other (``ResumeWithDirectoryDialog``
 * takes ``buildReconnectCommand`` from ``ReconnectSessionDialog``, which
 * offers ``SwitchHostDialog``), so hanging the shared label off any one of
 * them closes an import cycle.
 *
 * @param host - The host to render, e.g. ``{name: "mac-laptop", status:
 *   "online"}``.
 */
export function HostLabel({ host }: { host: Host }) {
  const isOnline = host.status === "online";
  return (
    <span className="flex items-center gap-2">
      {isPersistentHost(host) || host.name.toLowerCase().includes("cloud") ? (
        <MonitorCloudIcon className="size-4 text-muted-foreground" />
      ) : (
        <MonitorIcon className="size-4 text-muted-foreground" />
      )}
      <span className="text-ui">{host.name}</span>
      <span
        className={`inline-flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wider ${
          isOnline ? "text-green-600" : "text-muted-foreground"
        }`}
      >
        <span
          className={`inline-block size-1.5 rounded-full ${isOnline ? "bg-green-500" : "bg-muted-foreground"}`}
        />
        {hostStatusWord(host)}
      </span>
    </span>
  );
}
