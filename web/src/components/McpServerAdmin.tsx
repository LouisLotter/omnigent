import { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { getCurrentIsAdmin, resolveIdentity } from "@/lib/identity";
import {
  AdminRequestError,
  adminRequest,
  asRegistryService,
  authorizeRegistryService,
  listAdminServices,
  registryRequest,
  type AdminService,
  type AdminServicePreset,
  type AdminServiceSettings,
  type DiscoveredTool,
} from "@/lib/mcpRegistry";
import { cn } from "@/lib/utils";

type LoginType = AdminServiceSettings["auth"];

const LOGIN_TYPES: { value: LoginType; label: string; hint: string }[] = [
  { value: "none", label: "None", hint: "The server needs no login." },
  { value: "bearer", label: "API token", hint: "Each member pastes their own token." },
  {
    value: "oauth",
    label: "Sign in",
    hint: "Each member signs in with their own account (OAuth).",
  },
];

/** A catalog ID from a name: lower case letters, digits and hyphens, starting with a letter. */
export function serviceIdFrom(name: string): string {
  const slug = name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  const id = /^[a-z]/.test(slug) ? slug : slug ? `mcp-${slug}` : "";
  return id.slice(0, 40).replace(/-+$/, "");
}

function loginLabel(auth: AdminService["auth"]): string {
  return LOGIN_TYPES.find((type) => type.value === auth)?.label ?? auth;
}

/** Full settings for an update, from a service as listed, with *changes* applied. */
export function settingsFrom(
  service: AdminService,
  changes: Partial<AdminServiceSettings> = {},
): AdminServiceSettings {
  const auth = (
    service.auth === "none" || service.auth === "bearer" ? service.auth : "oauth"
  ) as LoginType;
  const oauth = service.oauth;
  return {
    title: service.title,
    description: service.description,
    url: service.url,
    auth,
    timeout: service.timeout,
    tools: service.tools,
    oauth:
      auth === "oauth" && oauth
        ? {
            ...(oauth.registration === "manual" ? { client_id: oauth.client_id } : {}),
            authorize_url: oauth.authorize_url,
            token_url: oauth.token_url,
            scopes: oauth.scopes,
            ...(oauth.resource ? { resource: oauth.resource } : {}),
          }
        : null,
    ...changes,
  };
}

/** Whether the signed-in person is an administrator, per the `/v1/me` probe. */
function useAdminIdentity(): boolean {
  const [admin, setAdmin] = useState(getCurrentIsAdmin);
  useEffect(() => {
    let live = true;
    void resolveIdentity().then(() => {
      if (live) setAdmin(getCurrentIsAdmin());
    });
    return () => {
      live = false;
    };
  }, []);
  return admin;
}

/** Settings → MCP for administrators: add, change and remove MCP servers. */
export function McpServerAdmin({ onChanged }: { onChanged?: () => void }) {
  // The server enforces the admin flag on every route; this only hides the controls.
  const isAdmin = useAdminIdentity();
  const [available, setAvailable] = useState<boolean | null>(null);
  const [services, setServices] = useState<AdminService[]>([]);
  const [presets, setPresets] = useState<AdminServicePreset[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState<{ service: AdminService | null; step: Step } | null>(null);
  const [removing, setRemoving] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const [listed, offered] = await Promise.all([
        listAdminServices(),
        adminRequest<{ data: AdminServicePreset[] }>("/presets").catch(() => ({ data: [] })),
      ]);
      setServices(listed);
      setPresets(offered.data);
      setAvailable(true);
      setError(null);
    } catch (e) {
      if (e instanceof AdminRequestError && (e.status === 404 || e.status === 403)) {
        setAvailable(false);
      } else {
        setError(e instanceof Error ? e.message : "Could not load MCP servers");
        setAvailable(true);
      }
    }
  }, []);

  useEffect(() => {
    if (isAdmin) void load();
  }, [isAdmin, load]);

  const changed = useCallback(async () => {
    await load();
    onChanged?.();
  }, [load, onChanged]);

  if (!isAdmin || available !== true) return null;
  return (
    <div className="space-y-3">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-medium">MCP servers</h3>
          <p className="text-sm text-muted-foreground">
            Servers you add here are available to every member at once. Each member connects their
            own account; logins are stored encrypted on the server and never reach computers.
          </p>
        </div>
        <Button size="sm" onClick={() => setOpen({ service: null, step: "details" })}>
          Add MCP server
        </Button>
      </div>
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
      <ul className="space-y-2">
        {services.map((service) => (
          <li
            key={service.id}
            className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-border p-3"
          >
            <div className="min-w-0">
              <p className="text-sm font-medium">
                {service.title}{" "}
                <span className="text-xs text-muted-foreground">({service.id})</span>
              </p>
              <p className="truncate text-xs text-muted-foreground">
                {service.url} · {loginLabel(service.auth)} ·{" "}
                {service.source === "config"
                  ? "from the server's configuration"
                  : service.draft
                    ? "setup not finished: no tools allowed yet"
                    : `${service.tools.length} tool${service.tools.length === 1 ? "" : "s"} allowed`}
              </p>
            </div>
            {service.editable && (
              <div className="flex gap-2">
                {removing === service.id ? (
                  <>
                    <span className="self-center text-xs text-muted-foreground">
                      Removes everyone's saved logins for it.
                    </span>
                    <Button
                      size="sm"
                      variant="destructive"
                      onClick={() =>
                        void (async () => {
                          try {
                            await adminRequest(`/services/${service.id}`, { method: "DELETE" });
                            setRemoving(null);
                            await changed();
                          } catch (e) {
                            setError(e instanceof Error ? e.message : "Could not remove it");
                          }
                        })()
                      }
                    >
                      Remove {service.title}
                    </Button>
                    <Button size="sm" variant="ghost" onClick={() => setRemoving(null)}>
                      Cancel
                    </Button>
                  </>
                ) : (
                  <>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() =>
                        setOpen({ service, step: service.draft ? "tools" : "details" })
                      }
                    >
                      {service.draft ? "Finish setup" : "Edit"}
                    </Button>
                    {!service.draft && (
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => setOpen({ service, step: "tools" })}
                      >
                        Tools
                      </Button>
                    )}
                    <Button size="sm" variant="ghost" onClick={() => setRemoving(service.id)}>
                      Remove
                    </Button>
                  </>
                )}
              </div>
            )}
          </li>
        ))}
      </ul>
      {open && (
        <McpServerDialog
          service={open.service}
          initialStep={open.step}
          presets={presets}
          onClose={() => setOpen(null)}
          onSaved={changed}
        />
      )}
    </div>
  );
}

type Step = "details" | "tools";

function McpServerDialog({
  service: initial,
  initialStep,
  presets,
  onClose,
  onSaved,
}: {
  service: AdminService | null;
  initialStep: Step;
  presets: AdminServicePreset[];
  onClose: () => void;
  onSaved: () => Promise<void>;
}) {
  const [service, setService] = useState<AdminService | null>(initial);
  const [step, setStep] = useState<Step>(initialStep);
  return (
    <Dialog open onOpenChange={(next) => !next && onClose()}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>
            {service === null
              ? "Add MCP server"
              : step === "tools"
                ? `Tools of ${service.title}`
                : `Edit ${service.title}`}
          </DialogTitle>
          <DialogDescription>
            {step === "details"
              ? "Name the server, give its URL and how members log in."
              : "Connect your own account, discover the server's tools, and choose which ones agents may use."}
          </DialogDescription>
        </DialogHeader>
        {step === "details" ? (
          <DetailsForm
            service={service}
            presets={presets}
            onCancel={onClose}
            onSaved={async (saved, isNew) => {
              setService(saved);
              await onSaved();
              if (isNew || saved.draft || saved.credentials_reset) setStep("tools");
              else onClose();
            }}
          />
        ) : (
          service && (
            <ToolsStep
              service={service}
              onDone={async () => {
                await onSaved();
                onClose();
              }}
            />
          )
        )}
      </DialogContent>
    </Dialog>
  );
}

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <label className="block space-y-1 text-sm">
      <span className="font-medium">{label}</span>
      {children}
      {hint && <span className="block text-xs text-muted-foreground">{hint}</span>}
    </label>
  );
}

function DetailsForm({
  service,
  presets,
  onCancel,
  onSaved,
}: {
  service: AdminService | null;
  presets: AdminServicePreset[];
  onCancel: () => void;
  onSaved: (service: AdminService, isNew: boolean) => Promise<void>;
}) {
  const creating = service === null;
  const [title, setTitle] = useState(service?.title ?? "");
  const [id, setId] = useState(service?.id ?? "");
  const idEdited = useRef(!creating);
  const [url, setUrl] = useState(service?.url ?? "");
  const [description, setDescription] = useState(service?.description ?? "");
  const [auth, setAuth] = useState<LoginType>(
    service && (service.auth === "none" || service.auth === "bearer")
      ? service.auth
      : service
        ? "oauth"
        : "bearer",
  );
  const manualClient = service?.oauth?.registration === "manual";
  const [clientId, setClientId] = useState(manualClient ? (service?.oauth?.client_id ?? "") : "");
  const [clientSecret, setClientSecret] = useState("");
  const [authorizeUrl, setAuthorizeUrl] = useState(service?.oauth?.authorize_url ?? "");
  const [tokenUrl, setTokenUrl] = useState(service?.oauth?.token_url ?? "");
  const [scopes, setScopes] = useState(service?.oauth?.scopes.join(" ") ?? "");
  const [advanced, setAdvanced] = useState(manualClient);
  const endpointsEdited = useRef(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function applyPreset(presetId: string) {
    const preset = presets.find((p) => p.id === presetId);
    if (!preset) return;
    setTitle(preset.title);
    setId(preset.id);
    idEdited.current = true;
    setUrl(preset.url);
    setDescription(preset.description);
    setAuth(preset.auth);
    setClientId(preset.oauth?.client_id ?? "");
    setScopes((preset.oauth?.scopes ?? []).join(" "));
  }

  function changeUrl(next: string) {
    setUrl(next);
    // A new address has its own authorization server: discover it again.
    if (!endpointsEdited.current) {
      setAuthorizeUrl("");
      setTokenUrl("");
      if (!creating) setScopes("");
    }
  }

  async function save() {
    setBusy(true);
    setError(null);
    const keepResource = service?.oauth?.resource && url === service.url;
    const settings: AdminServiceSettings = {
      title: title.trim(),
      description: description.trim(),
      url: url.trim(),
      auth,
      timeout: service?.timeout ?? 60,
      tools: service?.tools ?? [],
      oauth:
        auth === "oauth"
          ? {
              ...(clientId.trim() ? { client_id: clientId.trim() } : {}),
              ...(clientSecret ? { client_secret: clientSecret } : {}),
              ...(authorizeUrl.trim() && tokenUrl.trim()
                ? { authorize_url: authorizeUrl.trim(), token_url: tokenUrl.trim() }
                : {}),
              scopes: scopes.split(/\s+/).filter(Boolean),
              ...(keepResource ? { resource: service?.oauth?.resource ?? undefined } : {}),
            }
          : null,
    };
    try {
      const saved = creating
        ? await adminRequest<AdminService>("/services", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ id, ...settings }),
          })
        : await adminRequest<AdminService>(`/services/${service.id}`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(settings),
          });
      setClientSecret("");
      await onSaved(saved, creating);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not save the MCP server");
    } finally {
      setBusy(false);
    }
  }

  const changesLogin =
    !creating && service !== null && (url.trim() !== service.url || auth !== service.auth);
  return (
    <form
      className="space-y-3"
      onSubmit={(e) => {
        e.preventDefault();
        void save();
      }}
    >
      {creating && presets.length > 0 && (
        <Field label="Start from">
          <select
            aria-label="Start from"
            className="w-full rounded border border-border bg-background px-2 py-1.5 text-sm"
            defaultValue=""
            onChange={(e) => applyPreset(e.target.value)}
          >
            <option value="">A blank server</option>
            {presets.map((preset) => (
              <option key={preset.id} value={preset.id}>
                {preset.title}
              </option>
            ))}
          </select>
        </Field>
      )}
      <Field label="Name">
        <Input
          aria-label="Name"
          value={title}
          maxLength={120}
          onChange={(e) => {
            setTitle(e.target.value);
            if (!idEdited.current) setId(serviceIdFrom(e.target.value));
          }}
          required
        />
      </Field>
      {creating && (
        <Field label="ID" hint="Agents see tools as <id>__<tool>. It cannot change later.">
          <Input
            aria-label="ID"
            value={id}
            maxLength={40}
            pattern="[a-z][a-z0-9_\-]{0,39}"
            onChange={(e) => {
              idEdited.current = true;
              setId(e.target.value);
            }}
            required
          />
        </Field>
      )}
      <Field
        label="Server URL"
        hint="The server's streamable HTTP address, for example https://mcp.example.com/mcp."
      >
        <Input
          aria-label="Server URL"
          type="url"
          value={url}
          onChange={(e) => changeUrl(e.target.value)}
          placeholder="https://"
          required
        />
      </Field>
      <Field label="Description">
        <Input
          aria-label="Description"
          value={description}
          maxLength={1000}
          onChange={(e) => setDescription(e.target.value)}
        />
      </Field>
      <fieldset className="space-y-1 text-sm">
        <legend className="font-medium">Login</legend>
        <div role="radiogroup" aria-label="Login" className="flex flex-wrap gap-2">
          {LOGIN_TYPES.map((type) => (
            <Button
              key={type.value}
              type="button"
              size="sm"
              role="radio"
              aria-checked={auth === type.value}
              variant={auth === type.value ? "default" : "outline"}
              onClick={() => setAuth(type.value)}
            >
              {type.label}
            </Button>
          ))}
        </div>
        <p className="text-xs text-muted-foreground">
          {LOGIN_TYPES.find((type) => type.value === auth)?.hint}
          {auth === "oauth" &&
            " Omnigent finds the sign-in endpoints and registers itself where the server allows it."}
        </p>
      </fieldset>
      {auth === "oauth" && (
        <div className="space-y-2">
          <Button
            type="button"
            size="sm"
            variant="link"
            className="px-0"
            onClick={() => setAdvanced(!advanced)}
          >
            {advanced
              ? "Hide OAuth client settings"
              : "OAuth client settings (if the server needs them)"}
          </Button>
          {advanced && (
            <div className="space-y-2 rounded-md border border-border p-3">
              <Field label="Client ID" hint="Leave empty to register Omnigent automatically.">
                <Input
                  aria-label="Client ID"
                  value={clientId}
                  onChange={(e) => setClientId(e.target.value)}
                />
              </Field>
              <Field
                label="Client secret"
                hint={
                  service?.oauth?.has_client_secret
                    ? "Saved. Leave empty to keep it while the token URL and client ID stay the same."
                    : "Stored encrypted; never shown again."
                }
              >
                <Input
                  aria-label="Client secret"
                  type="password"
                  autoComplete="off"
                  value={clientSecret}
                  onChange={(e) => setClientSecret(e.target.value)}
                />
              </Field>
              <Field label="Authorization URL" hint="Leave both URLs empty to discover them.">
                <Input
                  aria-label="Authorization URL"
                  value={authorizeUrl}
                  onChange={(e) => {
                    endpointsEdited.current = true;
                    setAuthorizeUrl(e.target.value);
                  }}
                />
              </Field>
              <Field label="Token URL">
                <Input
                  aria-label="Token URL"
                  value={tokenUrl}
                  onChange={(e) => {
                    endpointsEdited.current = true;
                    setTokenUrl(e.target.value);
                  }}
                />
              </Field>
              <Field label="Scopes" hint="Separated by spaces; empty uses the server's own.">
                <Input
                  aria-label="Scopes"
                  value={scopes}
                  onChange={(e) => setScopes(e.target.value)}
                />
              </Field>
            </div>
          )}
        </div>
      )}
      {changesLogin && (
        <p role="note" className="text-sm text-amber-600 dark:text-amber-500">
          Changing the URL or the login type removes every member's saved login for this server.
        </p>
      )}
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
      <DialogFooter>
        <Button type="button" variant="ghost" onClick={onCancel}>
          Cancel
        </Button>
        <Button type="submit" disabled={busy || !title.trim() || !url.trim() || (creating && !id)}>
          {busy ? "Saving…" : creating ? "Save and continue" : "Save"}
        </Button>
      </DialogFooter>
    </form>
  );
}

function ToolsStep({ service, onDone }: { service: AdminService; onDone: () => Promise<void> }) {
  const [token, setToken] = useState("");
  const [tools, setTools] = useState<DiscoveredTool[] | null>(null);
  const [allowed, setAllowed] = useState<Set<string>>(new Set(service.tools));
  const [busy, setBusy] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const signIn = useRef<AbortController | null>(null);
  useEffect(() => () => signIn.current?.abort(), []);

  async function run(label: string, action: () => Promise<void>) {
    setBusy(label);
    setError(null);
    setNotice(null);
    try {
      await action();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setBusy(null);
    }
  }

  const discover = () =>
    run("discover", async () => {
      const result = await adminRequest<{ tools: DiscoveredTool[]; allowed: string[] }>(
        `/services/${service.id}/discover`,
        { method: "POST" },
      );
      setTools(result.tools);
      // A first discovery offers every tool; later ones keep the saved choice.
      setAllowed(new Set(result.allowed.length ? result.allowed : result.tools.map((t) => t.name)));
    });

  return (
    <div className="space-y-3">
      {service.auth === "bearer" && (
        <form
          className="flex gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            void run("token", async () => {
              await registryRequest(`/${service.id}/connection`, {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ token }),
              });
              setToken("");
              setNotice("Your token is saved as your own login.");
            });
          }}
        >
          <Input
            type="password"
            autoComplete="off"
            aria-label={`Your token for ${service.title}`}
            placeholder="Your API token"
            value={token}
            onChange={(e) => setToken(e.target.value)}
          />
          <Button size="sm" type="submit" disabled={busy !== null || !token.trim()}>
            Save my token
          </Button>
        </form>
      )}
      {service.auth === "oauth" && (
        <Button
          size="sm"
          variant="outline"
          disabled={busy !== null}
          onClick={() =>
            void run("sign-in", async () => {
              signIn.current?.abort();
              signIn.current = new AbortController();
              await authorizeRegistryService(asRegistryService(service), signIn.current.signal);
              setNotice("Signed in with your account.");
            })
          }
        >
          {busy === "sign-in" ? "Signing in…" : "Sign in with your account"}
        </Button>
      )}
      <div className="flex items-center gap-2">
        <Button size="sm" disabled={busy !== null} onClick={() => void discover()}>
          {busy === "discover" ? "Discovering…" : "Discover tools"}
        </Button>
        <span className="text-xs text-muted-foreground">
          Uses your own login to list what the server offers.
        </span>
      </div>
      {tools && (
        <fieldset className="max-h-64 space-y-1 overflow-y-auto rounded-md border border-border p-2">
          <legend className="px-1 text-sm font-medium">Tools agents may use</legend>
          {tools.length === 0 && (
            <p className="text-sm text-muted-foreground">The server offers no tools.</p>
          )}
          {tools.map((tool) => (
            <label key={tool.name} className="flex items-start gap-2 py-1 text-sm">
              <Checkbox
                aria-label={tool.name}
                checked={allowed.has(tool.name)}
                onCheckedChange={(checked) =>
                  setAllowed((current) => {
                    const next = new Set(current);
                    if (checked === true) next.add(tool.name);
                    else next.delete(tool.name);
                    return next;
                  })
                }
              />
              <span className="min-w-0">
                <span className="font-mono text-xs">{tool.name}</span>
                {tool.description && (
                  <span className="block text-xs text-muted-foreground">{tool.description}</span>
                )}
              </span>
            </label>
          ))}
        </fieldset>
      )}
      {notice && (
        <p role="status" className="text-sm">
          {notice}
        </p>
      )}
      {error && (
        <p role="alert" className={cn("text-sm text-destructive")}>
          {error}
        </p>
      )}
      <DialogFooter>
        <Button variant="ghost" onClick={() => void onDone()}>
          {tools ? "Cancel" : "Later"}
        </Button>
        <Button
          disabled={busy !== null || !tools || allowed.size === 0}
          onClick={() =>
            void run("save", async () => {
              await adminRequest(`/services/${service.id}`, {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(settingsFrom(service, { tools: [...allowed].sort() })),
              });
              await onDone();
            })
          }
        >
          Save allowed tools
        </Button>
      </DialogFooter>
    </div>
  );
}
