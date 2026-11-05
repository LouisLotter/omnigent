import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { CHANGING_HOSTS_REFRESH_MS, type Host } from "@/hooks/useHosts";
import { authenticatedFetch } from "@/lib/identity";
import { isPersistentHostChanging } from "@/lib/persistentHosts";
import { apiErrorFromResponse } from "@/lib/sessionsApi";
import { getMemberHosts } from "@/lib/teamApi";

/** Send a host request, throwing the server's error message on failure. */
async function requestHost<T>(path: string, init: RequestInit): Promise<T> {
  const res = await authenticatedFetch(path, init);
  if (!res.ok) throw await apiErrorFromResponse(res);
  return (await res.json()) as T;
}

function hostPath(hostId: string, suffix = ""): string {
  return `/v1/hosts/${encodeURIComponent(hostId)}${suffix}`;
}

/**
 * Team members' host lists, on an owner's or admin's page about them. Kept apart
 * from `["hosts"]`, which lists the caller's own hosts and which other code reads
 * as such.
 */
const MEMBER_HOSTS_KEY = "member-hosts";

function refreshHostLists(queryClient: ReturnType<typeof useQueryClient>) {
  return Promise.all([
    queryClient.invalidateQueries({ queryKey: ["hosts"] }),
    queryClient.invalidateQueries({ queryKey: [MEMBER_HOSTS_KEY] }),
  ]);
}

export interface CreatePersistentHostInput {
  name: string;
  provider: string | null;
  /** A team member to add it for, when an owner or admin adds one for them. */
  owner_user_id?: string;
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
    onSuccess: () => refreshHostLists(queryClient),
  });
}

/** Start a failed computer again, or wake an asleep one. */
export function useStartPersistentHost() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (hostId: string) =>
      requestHost<Host>(hostPath(hostId, "/start"), { method: "POST" }),
    onSuccess: () => refreshHostLists(queryClient),
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
      for (const key of ["hosts", MEMBER_HOSTS_KEY]) {
        queryClient.setQueriesData<Host[]>({ queryKey: [key] }, (hosts) =>
          hosts?.filter((host) => host.host_id !== hostId),
        );
      }
      return refreshHostLists(queryClient);
    },
  });
}

/**
 * A team member's hosts, for an owner's or admin's page about them. Starting,
 * deleting and adding computers refreshes it, and it polls while a computer is
 * changing, as the member's own list does.
 */
export function useMemberHosts(memberId: string) {
  return useQuery({
    queryKey: [MEMBER_HOSTS_KEY, memberId],
    queryFn: async (): Promise<Host[]> => {
      const result = await getMemberHosts(memberId);
      if (!result.ok) throw new Error(result.error);
      return result.hosts;
    },
    refetchInterval: (query) =>
      query.state.data?.some(isPersistentHostChanging) ? CHANGING_HOSTS_REFRESH_MS : false,
  });
}
