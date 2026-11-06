// Canonical model / effort label formatting for the composer, shared by the
// landing dialog and the in-session chat composer so both surfaces render the
// same label for the same model.
//
// Pure leaf module (no React, no store) so the landing screen, the chat page,
// the harness config controls, and the store can all depend on one source of
// truth without a circular import.

import { findNativeModelOption } from "@/lib/codexNativeModels";
import { fusionModelLabel, isFusionModelUid } from "@/lib/devinFusion";
import type { NativeModelOption } from "@/lib/types";

const DISPLAY_ONLY_CATALOG_PREFIXES = ["databricks-", "system.ai."] as const;

/** The native-catalog fields a model label is built from. A superset like
 *  {@link NativeModelOption} is assignable to this. */
export interface NativeModelLabelFields {
  id: string;
  model?: string;
  displayName?: string;
  isDefault?: boolean;
}

export function nativeModelLabel(option: NativeModelLabelFields): string {
  const label = option.displayName ?? option.model ?? option.id;
  // Some provider catalogs repeat the transport id as their display name.
  // Hide its mechanical namespace while preserving real advertised labels.
  const isTransportLabel = [option.id, option.model].some(
    (id) => id != null && (label === id || label === id.slice(id.indexOf("/") + 1)),
  );
  if (option.displayName != null && !isTransportLabel) return label;
  for (const prefix of DISPLAY_ONLY_CATALOG_PREFIXES) {
    if (label.startsWith(prefix)) return label.slice(prefix.length);
  }
  return label;
}

/** Hover text for a picker row: the exact model id it launches, which a family
 *  label such as "Opus" leaves out. None when the label already shows that id,
 *  or for a Fusion row, which launches the combo picked in its own selectors. */
export function nativeModelTooltip(
  option: NativeModelLabelFields & Pick<NativeModelOption, "fusion">,
): string | undefined {
  if (option.fusion !== undefined) return undefined;
  const id = option.model ?? option.id;
  return id === nativeModelLabel(option) ? undefined : id;
}

export function defaultModelLabel(options: readonly NativeModelLabelFields[]): string {
  const defaultOption = options.find((option) => option.isDefault);
  return defaultOption ? `Default (${nativeModelLabel(defaultOption)})` : "Default";
}

export function compactModelTriggerLabel(value: string): string {
  const withoutDefault = /^Default \((.*)\)$/.exec(value)?.[1] ?? value;
  return withoutDefault.replace(/\s*\((\d+(?:\.\d+)?[KMG]) context\)/gi, " $1");
}

/** Resolve display names across catalog spellings without losing context variants. */
export function findModelLabelOption(
  options: readonly NativeModelOption[],
  model: string | null,
): NativeModelOption | null {
  const raw = model?.trim();
  if (!raw) return null;
  const exact =
    options.find((option) => option.id === raw) ?? options.find((option) => option.model === raw);
  if (exact) return exact;
  const extendedContext = raw.toLowerCase().endsWith("[1m]");
  return findNativeModelOption(
    options.filter(
      (option) =>
        (option.model ?? option.id ?? "").toLowerCase().endsWith("[1m]") === extendedContext,
    ),
    raw,
  );
}

export function formatStatusModelLabel(
  model: string | null,
  codexModelOptions: readonly NativeModelOption[] = [],
): string | null {
  const raw = model?.trim();
  if (!raw) return null;
  if (isFusionModelUid(raw)) {
    const descriptor = codexModelOptions.find((candidate) => candidate.fusion)?.fusion;
    if (descriptor) return fusionModelLabel(descriptor, raw);
  }
  const option = findModelLabelOption(codexModelOptions, raw);
  return option ? nativeModelLabel(option) : raw;
}

/** Normalize a reasoning-effort value to its display label — the single place
 *  ``xhigh`` becomes ``xHigh``. Any other value is capitalized. */
export function normalizeEffortLabel(effort: string): string {
  if (effort.toLowerCase() === "xhigh") return "xHigh";
  return effort.charAt(0).toUpperCase() + effort.slice(1);
}

/** Display label for a reasoning-effort value, or ``null`` when unset. */
export function formatStatusEffortLabel(effort: string | null): string | null {
  if (!effort) return null;
  return normalizeEffortLabel(effort);
}

/**
 * The effort to show as the one a session runs at, or ``null`` when unknown.
 * A harness that reports its effort never has its request shown in its place.
 *
 * @param reported - Effort the harness reports it runs at, or ``null``.
 * @param requested - The session's requested effort, or ``null``.
 * @param reportsEffort - Whether the harness reports its effective effort.
 */
export function displayedSessionEffort(
  reported: string | null,
  requested: string | null,
  reportsEffort: boolean,
): string | null {
  return reported ?? (reportsEffort ? null : requested);
}

/**
 * Hover-summary rows for a session's effort: the effective effort, its source,
 * and the request when it differs. For a harness that reports its effort, a
 * missing report reads "Unknown"; other harnesses fall back to the request.
 *
 * @param reported - Effort the harness reports it runs at, or ``null``.
 * @param requested - The session's requested effort, or ``null``.
 * @param harnessLabel - Harness display name for the source row, e.g. ``"Claude Code"``.
 * @param reportsEffort - Whether the harness reports its effective effort.
 * @returns Rows such as ``Effort: xHigh`` and ``Effort source: Reported by Claude Code``.
 */
export function effortSummaryRows(
  reported: string | null,
  requested: string | null,
  harnessLabel: string | null,
  reportsEffort: boolean,
): { label: string; value: string }[] {
  const effective = displayedSessionEffort(reported, requested, reportsEffort);
  if (effective === null && !reportsEffort) return [];
  const rows = [
    { label: "Effort", value: effective === null ? "Unknown" : normalizeEffortLabel(effective) },
  ];
  if (reported) {
    rows.push({ label: "Effort source", value: `Reported by ${harnessLabel ?? "the harness"}` });
  }
  if (requested && requested !== effective) {
    rows.push({ label: "Requested effort", value: normalizeEffortLabel(requested) });
  }
  return rows;
}

/**
 * Compose the current model and effort for the composer status tray.
 *
 * @param model - Model override or bound model id.
 * @param effort - Current reasoning effort override, if any.
 * @returns Compact label such as ``"gpt-5.5 xHigh"``, or ``null`` when neither is known.
 */
export function formatModelEffortStatusLabel(
  model: string | null,
  effort: string | null,
  codexModelOptions: readonly NativeModelOption[] = [],
): string | null {
  const modelLabel = formatStatusModelLabel(model, codexModelOptions);
  const effortLabel = formatStatusEffortLabel(effort);
  const parts = [modelLabel, effortLabel].filter((p): p is string => p != null && p.length > 0);
  return parts.length > 0 ? parts.join(" ") : null;
}
