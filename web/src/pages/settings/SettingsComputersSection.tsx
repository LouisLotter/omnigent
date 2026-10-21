import { useState } from "react";
import { Loader2Icon, PlusIcon } from "lucide-react";

import { AddComputerDialog } from "@/components/computers/AddComputerDialog";
import { ManageComputerLink } from "@/components/computers/ComputerStatus";
import { DeleteComputerDialog } from "@/components/computers/DeleteComputerDialog";
import { Button } from "@/components/ui/button";
import { useHosts, type Host } from "@/hooks/useHosts";
import { useStartPersistentHost } from "@/hooks/usePersistentHosts";
import { useServerInfo } from "@/lib/CapabilitiesContext";
import { persistentHostProviders } from "@/lib/capabilities";
import {
  describePersistentHostState,
  isPersistentHost,
  isPersistentHostChanging,
  isPersistentHostFailed,
  persistentHostManageUrl,
  persistentHostRefusal,
} from "@/lib/persistentHosts";

/**
 * Settings → Computers: the person's persistent cloud computers with their
 * state, setup link, "Start again" for a failed start, and delete. Shown only
 * when the server has a provider that creates computers.
 */
export function SettingsComputersSection() {
  const providers = persistentHostProviders(useServerInfo());
  const hosts = useHosts();
  const [addOpen, setAddOpen] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<Host | null>(null);
  const computers = (hosts.data ?? []).filter(isPersistentHost);

  return (
    <section>
      <div className="flex flex-wrap items-center justify-between gap-4">
        <h1 className="settings-page-title text-2xl font-semibold">Computers</h1>
        {providers.length > 0 && (
          <Button
            size="sm"
            className="gap-1.5"
            onClick={() => setAddOpen(true)}
            data-testid="add-computer"
          >
            <PlusIcon className="size-4" />
            Add computer
          </Button>
        )}
      </div>
      <p className="mt-1 text-ui text-muted-foreground max-md:hidden">
        Cloud computers keep their files between sessions. They sleep when idle and wake when you
        send them work.
      </p>
      <div className="mt-6">
        {hosts.isError && !hosts.data ? (
          <p role="alert" className="text-ui">
            Couldn't load your computers.{" "}
            <Button variant="link" onClick={() => void hosts.refetch()}>
              Retry
            </Button>
          </p>
        ) : hosts.data === undefined ? (
          <p className="text-ui text-muted-foreground">Loading computers…</p>
        ) : computers.length === 0 ? (
          <p className="text-ui text-muted-foreground" data-testid="computers-empty">
            No computers yet.
          </p>
        ) : (
          <ul className="divide-y divide-border rounded-xl border border-border bg-card">
            {computers.map((host) => (
              <ComputerRow key={host.host_id} host={host} onDelete={() => setDeleteTarget(host)} />
            ))}
          </ul>
        )}
      </div>
      <AddComputerDialog open={addOpen} onOpenChange={setAddOpen} providers={providers} />
      <DeleteComputerDialog host={deleteTarget} onClose={() => setDeleteTarget(null)} />
    </section>
  );
}

function ComputerRow({ host, onDelete }: { host: Host; onDelete: () => void }) {
  const start = useStartPersistentHost();
  const state = describePersistentHostState(host);
  const refusal = persistentHostRefusal(host);
  const manageUrl = persistentHostManageUrl(host);
  return (
    <li
      className="flex flex-wrap items-center justify-between gap-x-6 gap-y-3 p-4"
      data-testid={`computer-${host.host_id}`}
    >
      <div className="flex min-w-0 flex-1 flex-col gap-1">
        <span className="truncate text-ui font-medium">{host.name}</span>
        {state && (
          <span
            className="flex items-center gap-1.5 text-xs text-muted-foreground"
            data-testid="computer-state"
          >
            {isPersistentHostChanging(host) && (
              <Loader2Icon aria-hidden className="size-3 shrink-0 animate-spin" />
            )}
            <span className={isPersistentHostFailed(host) ? "text-destructive" : undefined}>
              {state.label}
            </span>
            {state.detail && <span>· {state.detail}</span>}
          </span>
        )}
        {refusal && (
          <span
            className="text-xs text-amber-600 dark:text-amber-500"
            data-testid="computer-refusal"
          >
            {refusal.message}
          </span>
        )}
        {start.error && (
          <span role="alert" className="text-xs text-destructive">
            {start.error.message}
          </span>
        )}
      </div>
      <div className="flex shrink-0 flex-wrap items-center gap-2">
        {manageUrl && <ManageComputerLink url={manageUrl} className="px-2 text-ui" />}
        {isPersistentHostFailed(host) && (
          <Button
            variant="outline"
            size="sm"
            loading={start.isPending}
            onClick={() => start.mutate(host.host_id)}
          >
            Start again
          </Button>
        )}
        <Button variant="ghost" size="sm" onClick={onDelete}>
          Delete
        </Button>
      </div>
    </li>
  );
}
