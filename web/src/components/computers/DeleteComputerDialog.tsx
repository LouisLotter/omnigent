import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import type { Host } from "@/hooks/useHosts";
import { useDeletePersistentHost } from "@/hooks/usePersistentHosts";
import { persistentHostManageUrl } from "@/lib/persistentHosts";
import { ManageComputerLink } from "./ComputerStatus";

/**
 * Confirms deleting a computer. Deletion also removes its files, so the
 * dialog first points the person to sign out on the computer's setup page.
 * With *ownerName*, an owner or admin deletes a team member's computer, and
 * the sign-out is the member's to do.
 */
export function DeleteComputerDialog({
  host,
  onClose,
  ownerName,
}: {
  /** The computer to delete; null keeps the dialog closed. */
  host: Host | null;
  onClose: () => void;
  /** The team member whose computer it is; the caller's own when omitted. */
  ownerName?: string;
}) {
  const remove = useDeletePersistentHost();
  const manageUrl = host ? persistentHostManageUrl(host) : null;

  function handleOpenChange(open: boolean) {
    if (open || remove.isPending) return;
    remove.reset();
    onClose();
  }

  function confirm() {
    if (!host) return;
    remove.mutate(host.host_id, {
      onSuccess: () => {
        remove.reset();
        onClose();
      },
    });
  }

  return (
    <Dialog open={host !== null} onOpenChange={handleOpenChange}>
      <DialogContent showCloseButton={!remove.isPending} data-testid="delete-computer-dialog">
        <DialogHeader>
          <DialogTitle>Delete {host?.name}?</DialogTitle>
          <DialogDescription>
            Deleting removes the computer and its files. This cannot be undone.
          </DialogDescription>
        </DialogHeader>
        <p className="text-ui" data-testid="delete-computer-sign-out">
          {ownerName !== undefined ? (
            `The sign-ins on it are ${ownerName}'s. Ask ${ownerName} to sign out of them first.`
          ) : manageUrl ? (
            <>
              Before deleting, sign out of the accounts you use on this computer: open{" "}
              <ManageComputerLink url={manageUrl} />.
            </>
          ) : (
            "Before deleting, sign out of the accounts you use on this computer."
          )}
        </p>
        {remove.error && (
          <p role="alert" className="text-ui text-destructive" data-testid="delete-computer-error">
            {remove.error.message}
          </p>
        )}
        <DialogFooter>
          <Button
            variant="ghost"
            disabled={remove.isPending}
            onClick={() => handleOpenChange(false)}
          >
            Cancel
          </Button>
          <Button
            variant="destructive"
            loading={remove.isPending}
            onClick={confirm}
            data-testid="delete-computer-confirm"
          >
            Delete
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
