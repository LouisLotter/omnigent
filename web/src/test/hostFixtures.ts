import type { Host, PersistentHostStatus } from "@/hooks/useHosts";

/**
 * A persistent managed host ("computer") as `GET /v1/hosts` lists it. Asleep
 * unless `status.state` says otherwise; online only when awake.
 */
export function computerHost(
  status: Partial<PersistentHostStatus> = {},
  host: Partial<Host> = {},
): Host {
  const state = status.state ?? "asleep";
  return {
    host_id: "host_computer",
    name: "research-box",
    owner: "me",
    status: state === "awake" ? "online" : "offline",
    sandbox_provider: "cloudflare",
    lifecycle: "persistent",
    persistent_status: {
      state,
      wake_stage: null,
      failure: null,
      admission: null,
      manage_url: null,
      provider: "cloudflare",
      ...status,
    },
    ...host,
  };
}
