import { useMutation, useQueryClient } from "@tanstack/react-query";

import type { Host } from "@/hooks/useHosts";
import { authenticatedFetch } from "@/lib/identity";
import { apiErrorFromResponse } from "@/lib/sessionsApi";

/** Send a host request, throwing the server's error message on failure. */
async function requestHost<T>(path: string, init: RequestInit): Promise<T> {
  const res = await authenticatedFetch(path, init);
  if (!res.ok) throw await apiErrorFromResponse(res);
  return (await res.json()) as T;
}

function hostPath(hostId: string, suffix = ""): string {
  return `/v1/hosts/${encodeURIComponent(hostId)}${suffix}`;
}

export interface CreatePersistentHostInput {
  name: string;
  provider: string | null;
}

/** Create a computer; it starts in the background and lists as waking. */
export function useCreatePersistentHost() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (input: CreatePersistentHostInput) =>
      requestHost<Host>("/v1/hosts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(input),
      }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["hosts"] }),
  });
}

/** Start a failed computer again, or wake an asleep one. */
export function useStartPersistentHost() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (hostId: string) =>
      requestHost<Host>(hostPath(hostId, "/start"), { method: "POST" }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["hosts"] }),
  });
}

/** Delete a computer and its files on the provider. */
export function useDeletePersistentHost() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (hostId: string) =>
      requestHost<{ host_id: string; deleted: boolean }>(hostPath(hostId), { method: "DELETE" }),
    onSuccess: (_result, hostId) => {
      // Drop the row now rather than after the refetch lands.
      queryClient.setQueriesData<Host[]>({ queryKey: ["hosts"] }, (hosts) =>
        hosts?.filter((host) => host.host_id !== hostId),
      );
      return queryClient.invalidateQueries({ queryKey: ["hosts"] });
    },
  });
}
