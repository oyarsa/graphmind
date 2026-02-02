/**
 * Evaluation settings management for the Paper Explorer.
 */

/** Settings interface for evaluation parameters */
export interface EvaluationSettings {
  llm_model: "gpt-4o" | "gpt-4o-mini" | "gemini-2.0-flash";
  k_refs: number;
  recommendations: number;
  related: number;
  filter_by_date: boolean;
}

export const DEFAULT_SETTINGS: EvaluationSettings = {
  llm_model: "gpt-4o-mini",
  k_refs: 20,
  recommendations: 30,
  related: 5,
  filter_by_date: true,
};

export const SETTINGS_STORAGE_KEY = "paper-explorer-evaluation-settings";

export function loadSettings(): EvaluationSettings {
  try {
    const stored = localStorage.getItem(SETTINGS_STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored) as Partial<EvaluationSettings>;
      return { ...DEFAULT_SETTINGS, ...parsed };
    }
  } catch (e) {
    console.warn("Failed to load settings:", e);
  }
  return { ...DEFAULT_SETTINGS };
}

export function saveSettings(settings: EvaluationSettings): void {
  try {
    localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings));
  } catch (e) {
    console.warn("Failed to save settings:", e);
  }
}

/**
 * Apply settings to form elements.
 */
export function applySettingsToForm(settings: EvaluationSettings): void {
  const llmModel = document.getElementById(
    "settings-llm-model",
  ) as HTMLSelectElement | null;
  const kRefs = document.getElementById("settings-k-refs") as HTMLInputElement | null;
  const recommendations = document.getElementById(
    "settings-recommendations",
  ) as HTMLInputElement | null;
  const related = document.getElementById(
    "settings-related",
  ) as HTMLInputElement | null;
  const filterByDate = document.getElementById(
    "settings-filter-by-date",
  ) as HTMLInputElement | null;

  if (llmModel) llmModel.value = settings.llm_model;
  if (kRefs) kRefs.value = settings.k_refs.toString();
  if (recommendations) recommendations.value = settings.recommendations.toString();
  if (related) related.value = settings.related.toString();
  if (filterByDate) filterByDate.checked = settings.filter_by_date;
}

/**
 * Get settings from form elements.
 */
export function getSettingsFromForm(): EvaluationSettings {
  const llmModel = document.getElementById(
    "settings-llm-model",
  ) as HTMLSelectElement | null;
  const kRefs = document.getElementById("settings-k-refs") as HTMLInputElement | null;
  const recommendations = document.getElementById(
    "settings-recommendations",
  ) as HTMLInputElement | null;
  const related = document.getElementById(
    "settings-related",
  ) as HTMLInputElement | null;
  const filterByDate = document.getElementById(
    "settings-filter-by-date",
  ) as HTMLInputElement | null;

  return {
    llm_model: (llmModel?.value ?? "gpt-4o-mini") as EvaluationSettings["llm_model"],
    k_refs: parseInt(kRefs?.value ?? "20", 10),
    recommendations: parseInt(recommendations?.value ?? "30", 10),
    related: parseInt(related?.value ?? "5", 10),
    filter_by_date: filterByDate?.checked ?? true,
  };
}
