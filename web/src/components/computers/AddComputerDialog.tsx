import { type FormEvent, useId, useState } from "react";

import { Label } from "@/components/scheduled/Label";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { Host } from "@/hooks/useHosts";
import { useCreatePersistentHost } from "@/hooks/usePersistentHosts";
import { sandboxProviderName } from "@/lib/capabilities";

/** The server rejects longer computer names. */
export const COMPUTER_NAME_MAX_LENGTH = 64;

/**
 * Names a new computer and creates it on a persistent-host provider. The
 * provider field appears only when the server offers more than one.
 */
export function AddComputerDialog({
  open,
  onOpenChange,
  providers,
  onCreated,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Providers that create computers, from `/v1/info`; the first is the default. */
  providers: string[];
  onCreated?: (host: Host) => void;
}) {
  const nameId = useId();
  const providerId = useId();
  const [name, setName] = useState("");
  const [provider, setProvider] = useState<string | null>(null);
  const create = useCreatePersistentHost();
  const trimmedName = name.trim();
  const nameTooLong = trimmedName.length > COMPUTER_NAME_MAX_LENGTH;
  const selectedProvider = provider ?? providers[0] ?? null;
  const canSubmit = trimmedName !== "" && !nameTooLong && !create.isPending;

  function resetForm() {
    setName("");
    setProvider(null);
    create.reset();
  }

  function handleOpenChange(next: boolean) {
    if (create.isPending) return;
    if (!next) resetForm();
    onOpenChange(next);
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    if (!canSubmit) return;
    create.mutate(
      { name: trimmedName, provider: selectedProvider },
      {
        onSuccess: (host) => {
          resetForm();
          onOpenChange(false);
          onCreated?.(host);
        },
      },
    );
  }

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent showCloseButton={!create.isPending} data-testid="add-computer-dialog">
        <form onSubmit={submit} className="flex flex-col gap-4">
          <DialogHeader>
            <DialogTitle>Add computer</DialogTitle>
            <DialogDescription>
              A cloud computer keeps its files between sessions. It sleeps when idle and wakes when
              you send it work.
            </DialogDescription>
          </DialogHeader>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor={nameId}>Name</Label>
            <Input
              id={nameId}
              autoFocus
              value={name}
              onChange={(event) => setName(event.target.value)}
              aria-invalid={nameTooLong || undefined}
              data-testid="add-computer-name"
            />
            {nameTooLong && (
              <p role="alert" className="text-xs text-destructive">
                Use {COMPUTER_NAME_MAX_LENGTH} characters or fewer.
              </p>
            )}
          </div>
          {providers.length > 1 && (
            <div className="flex flex-col gap-1.5">
              <Label htmlFor={providerId}>Provider</Label>
              <Select value={selectedProvider ?? undefined} onValueChange={setProvider}>
                <SelectTrigger
                  id={providerId}
                  className="w-full"
                  data-testid="add-computer-provider"
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {providers.map((option) => (
                    <SelectItem key={option} value={option}>
                      {sandboxProviderName(option)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
          {create.error && (
            <p role="alert" className="text-ui text-destructive" data-testid="add-computer-error">
              {create.error.message}
            </p>
          )}
          <DialogFooter>
            <Button
              type="button"
              variant="ghost"
              disabled={create.isPending}
              onClick={() => handleOpenChange(false)}
            >
              Cancel
            </Button>
            <Button
              type="submit"
              loading={create.isPending}
              disabled={!canSubmit}
              data-testid="add-computer-submit"
            >
              Add computer
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
