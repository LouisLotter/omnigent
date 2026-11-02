import { authenticatedFetch } from "./identity";
import { withBasePath } from "./basePath";
import { databricksConnectUrl } from "./databricksIntegration";

export interface RegistryService {
  id: string;
  title: string;
  description: string;
  auth: "none" | "bearer" | "oauth" | "github" | "databricks";
  connected: boolean;
  /** The provider rejected the saved login; absent from older servers. */
  needs_reconnect?: boolean;
  tools: string[];
  connect_provider: string;
}

export async function registryRequest<T>(path = "", init?: RequestInit): Promise<T> {
  const response = await authenticatedFetch(`/v1/mcp-registry/services${path}`, init);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : "MCP registry request failed");
  }
  return response.json() as Promise<T>;
}

export function authorizeRegistryService(
  service: RegistryService,
  signal: AbortSignal,
  workspace?: string,
) {
  if (service.auth === "databricks" && !workspace?.trim()) {
    return Promise.reject(new Error("Enter your Databricks workspace URL."));
  }
  const returnTo = withBasePath("/settings/mcp");
  const url =
    service.auth === "databricks"
      ? databricksConnectUrl(workspace!.trim(), returnTo)
      : withBasePath(
          `/v1/connections/${encodeURIComponent(service.connect_provider)}/connect?return_to=${encodeURIComponent(returnTo)}`,
        );
  const popup = window.open("about:blank", "_blank", "popup,width=600,height=720");
  if (!popup) return Promise.reject(new Error("Allow pop-ups to connect this account."));
  // The provider page must not be able to navigate the session window.
  popup.opener = null;
  popup.location.href = url;
  return new Promise<void>((resolve, reject) => {
    const finish = (error?: string) => {
      window.clearInterval(interval);
      window.clearTimeout(timeout);
      signal.removeEventListener("abort", cancel);
      popup.close();
      if (error) reject(new Error(error));
      else resolve();
    };
    const cancel = () => finish("Sign-in cancelled. Try again when ready.");
    const interval = window.setInterval(() => {
      if (popup.closed) {
        cancel();
        return;
      }
      try {
        const location = new URL(popup.location.href);
        if (location.origin !== window.location.origin || location.pathname !== returnTo) return;
        const status = location.searchParams.get(service.connect_provider);
        if (status === "connected") finish();
        else if (status === "error") finish("Sign-in failed. Please try again.");
      } catch {
        // The provider's cross-origin page is unreadable until it redirects back.
      }
    }, 500);
    const timeout = window.setTimeout(
      () => finish("Sign-in timed out. Please try again."),
      300_000,
    );
    signal.addEventListener("abort", cancel, { once: true });
    if (signal.aborted) cancel();
  });
}

/** OAuth settings of a service, as administrators see them; never a secret. */
export interface AdminServiceOAuth {
  authorize_url: string;
  token_url: string;
  client_id: string;
  scopes: string[];
  resource: string | null;
  has_client_secret: boolean;
  registration?: "manual" | "dynamic" | null;
  token_endpoint_auth?: TokenEndpointAuth;
}

/** How an OAuth client authenticates at the token endpoint. */
export type TokenEndpointAuth = "none" | "client_secret_post" | "client_secret_basic";

/** A catalog service in the administrators' list. */
export interface AdminService {
  id: string;
  title: string;
  description: string;
  url: string;
  auth: RegistryService["auth"];
  timeout: number;
  tools: string[];
  /** ``admin``: added in Settings; ``config``: from the server's registry file. */
  source: "admin" | "config";
  editable: boolean;
  oauth: AdminServiceOAuth | null;
  /** No tools allowed yet: only its creator sees it, to finish setting it up. */
  draft?: boolean;
  credentials_reset?: boolean;
}

/** What the Add/Edit MCP server form sends. ``client_secret`` is write-only. */
export interface AdminServiceSettings {
  title: string;
  description: string;
  url: string;
  auth: "none" | "bearer" | "oauth";
  timeout: number;
  tools: string[];
  oauth?: {
    client_id?: string;
    client_secret?: string;
    /** Sent with ``client_id``; without it the server keeps the same client's saved method. */
    token_endpoint_auth?: TokenEndpointAuth;
    authorize_url?: string;
    token_url?: string;
    scopes?: string[];
    resource?: string;
  } | null;
}

export interface AdminServicePreset {
  id: string;
  title: string;
  description: string;
  url: string;
  auth: "none" | "bearer" | "oauth";
  tools: string[];
  timeout: number;
  oauth?: { client_id?: string | null; scopes?: string[] } | null;
}

export interface DiscoveredTool {
  name: string;
  title?: string | null;
  description: string;
}

/** Thrown for an admin request; ``status`` 404 means the server has no admin services. */
export class AdminRequestError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

export async function adminRequest<T>(path = "", init?: RequestInit): Promise<T> {
  const response = await authenticatedFetch(`/v1/mcp-registry/admin${path}`, init);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const detail =
      typeof body.detail === "string"
        ? body.detail
        : Array.isArray(body.detail)
          ? "Check the highlighted fields and try again."
          : "MCP server request failed";
    throw new AdminRequestError(detail, response.status);
  }
  return response.json() as Promise<T>;
}

export async function listAdminServices(): Promise<AdminService[]> {
  const services: AdminService[] = [];
  let cursor: string | null = null;
  do {
    const page: { data: AdminService[]; next_cursor?: string | null } =
      // eslint-disable-next-line no-await-in-loop -- Each page needs the previous cursor.
      await adminRequest(`/services${cursor ? `?after=${encodeURIComponent(cursor)}` : ""}`);
    services.push(...page.data);
    cursor = page.next_cursor ?? null;
  } while (cursor);
  return services;
}

/** A registry entry for a service, so the shared sign-in flow can connect to it. */
export function asRegistryService(service: AdminService): RegistryService {
  return {
    id: service.id,
    title: service.title,
    description: service.description,
    auth: service.auth,
    connected: false,
    tools: service.tools,
    connect_provider: service.oauth ? `mcp-${service.id}` : service.auth,
  };
}
