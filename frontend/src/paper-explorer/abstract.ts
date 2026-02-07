/**
 * Abstract evaluation functionality for the Paper Explorer.
 */

import { z } from "zod/v4";
import { cleanKeyword } from "../util";
import { renderLatex } from "./helpers";
import { AbstractEvaluationResponse, AbstractEvaluationResponseSchema } from "./model";
import { AbstractEvaluationParams } from "./services";
import { parseStoredJson } from "./storage";

/**
 * Setup Abstract tab functionality including form submission and cache management.
 */
export function setupAbstractTab(
  _abstractEvaluator: unknown,
  evaluateAbstract: (params: AbstractEvaluationParams) => Promise<void>,
): void {
  setupAbstractForm(evaluateAbstract);
  setupAbstractClearCache();
}

/**
 * Setup the abstract evaluation form submission.
 */
function setupAbstractForm(
  evaluateAbstract: (params: AbstractEvaluationParams) => Promise<void>,
): void {
  const form = document.getElementById(
    "abstract-evaluation-form",
  ) as HTMLFormElement | null;
  if (!form) return;

  form.addEventListener("submit", e => {
    e.preventDefault();

    const titleElement = document.getElementById(
      "abstract-title",
    ) as HTMLTextAreaElement | null;
    const abstractElement = document.getElementById(
      "abstract-text",
    ) as HTMLTextAreaElement | null;
    const recommendationsElement = document.getElementById(
      "abstract-recommendations",
    ) as HTMLInputElement | null;
    const relatedElement = document.getElementById(
      "abstract-related",
    ) as HTMLInputElement | null;
    const llmModelElement = document.getElementById(
      "abstract-llm-model",
    ) as HTMLSelectElement | null;

    if (
      !titleElement ||
      !abstractElement ||
      !recommendationsElement ||
      !relatedElement ||
      !llmModelElement
    ) {
      console.error("Required form elements not found");
      return;
    }

    const title = titleElement.value.trim();
    const abstract = abstractElement.value.trim();

    if (!title || !abstract) {
      showAbstractError("Please provide both title and abstract.");
      return;
    }

    const params: AbstractEvaluationParams = {
      title,
      abstract,
      recommendations: parseInt(recommendationsElement.value) || 20,
      related: parseInt(relatedElement.value) || 5,
      llm_model: llmModelElement.value || "gpt-4o-mini",
    };

    void evaluateAbstract(params);
  });
}

/**
 * Setup clear cache button for abstract evaluations.
 */
function setupAbstractClearCache(): void {
  const clearButton = document.getElementById("clear-abstract-cache");
  if (!clearButton) return;

  clearButton.addEventListener("click", () => {
    if (!confirm("Are you sure you want to clear all previous abstract evaluations?")) {
      return;
    }

    const keys = Object.keys(localStorage);
    const abstractKeys = keys.filter(key =>
      key.startsWith("abstract-evaluation-cache-"),
    );

    abstractKeys.forEach(key => localStorage.removeItem(key));
    localStorage.removeItem("abstract-evaluations-list");

    loadPreviousAbstractEvaluations();
  });
}

/**
 * Load and display previous abstract evaluations.
 */
export function loadPreviousAbstractEvaluations(): void {
  const container = document.getElementById("abstract-results-container");
  const noResults = document.getElementById("abstract-no-results");

  if (!container || !noResults) return;

  const evaluationsList = localStorage.getItem("abstract-evaluations-list");
  const evaluationIds =
    parseStoredJson(
      evaluationsList,
      z.array(z.string()),
      "abstract-evaluations-list",
    ) ?? [];

  container.innerHTML = "";

  if (evaluationIds.length === 0) {
    container.classList.add("hidden");
    noResults.classList.remove("hidden");
    return;
  }

  noResults.classList.add("hidden");
  container.classList.remove("hidden");

  const evaluations: AbstractEvaluationResponse[] = [];

  for (const id of evaluationIds) {
    const key = `abstract-evaluation-cache-${id}`;
    const cached = localStorage.getItem(key);
    const evaluation = parseStoredJson(cached, AbstractEvaluationResponseSchema, key);
    if (evaluation) {
      evaluations.push(evaluation);
    }
  }

  evaluations.sort((a, b) => b.id.localeCompare(a.id));

  evaluations.forEach(evaluation => {
    const card = createAbstractEvaluationCard(evaluation);
    container.appendChild(card);
  });
}

/**
 * Create a card element for displaying an abstract evaluation result.
 */
function createAbstractEvaluationCard(
  evaluation: AbstractEvaluationResponse,
): HTMLElement {
  const card = document.createElement("div");
  card.className =
    "rounded-lg border border-gray-300 bg-gray-50/50 p-4 transition-all " +
    "duration-200 hover:border-teal-500/50 dark:border-gray-700 dark:bg-gray-800/50";

  const rating = evaluation.label;
  const ratingDisplay = `${rating}/5`;
  const noveltyColor =
    rating <= 2
      ? "text-red-600 dark:text-red-400"
      : rating === 3
        ? "text-yellow-600 dark:text-yellow-400"
        : "text-green-600 dark:text-green-400";

  card.innerHTML = `
    <div class="cursor-pointer" onclick="window.location.href='/graphmind/pages/abstract-detail.html?id=${evaluation.id}'">
      <h4 class="mb-2 line-clamp-2 font-semibold text-gray-900 dark:text-gray-100">
        ${renderLatex(evaluation.title)}
      </h4>
      <p class="mb-3 line-clamp-3 text-sm text-gray-600 dark:text-gray-400">
        ${evaluation.abstract.substring(0, 150)}${evaluation.abstract.length > 150 ? "..." : ""}
      </p>
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-2">
          <span class="text-xs font-medium ${noveltyColor}">
            Novelty: ${ratingDisplay}
          </span>
        </div>
        <div class="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-500">
          <span>${evaluation.keywords.length} keywords</span>
          <span>•</span>
          <span>${evaluation.related.length} related</span>
        </div>
      </div>
      <div class="mt-2 flex flex-wrap gap-1">
        ${evaluation.keywords
          .slice(0, 3)
          .map(
            keyword => `
          <span class="inline-block rounded-md bg-teal-100/70 px-2 py-0.5 text-xs text-teal-800
                 dark:bg-teal-900/30 dark:text-teal-300">
            ${cleanKeyword(keyword)}
          </span>
        `,
          )
          .join("")}
        ${
          evaluation.keywords.length > 3
            ? `
          <span class="text-xs text-gray-500 dark:text-gray-500">
            +${evaluation.keywords.length - 3} more
          </span>
        `
            : ""
        }
      </div>
    </div>
  `;

  return card;
}

/**
 * Show progress modal for abstract evaluation.
 */
export function showAbstractEvaluationModal(title: string): void {
  hideAbstractEvaluationModal();

  const modal = document.createElement("div");
  modal.id = "abstract-evaluation-modal";
  modal.className =
    "fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm";

  modal.innerHTML = `
    <div class="mx-4 w-96 rounded-lg bg-white p-6 shadow-xl dark:bg-gray-800">
      <div class="text-center">
        <h3 class="mb-2 text-lg font-semibold text-gray-900 dark:text-white">
          Evaluating Abstract
        </h3>
        <p class="mb-4 text-sm text-gray-600 dark:text-gray-400 line-clamp-2">
          ${renderLatex(title)}
        </p>

        <!-- Progress Bar -->
        <div class="mb-4 w-full bg-gray-200 rounded-full h-2 dark:bg-gray-700">
          <div id="abstract-progress-bar" class="bg-teal-600 h-2 rounded-full transition-all duration-300"
               style="width: 0%">
          </div>
        </div>

        <!-- Progress Text -->
        <div class="mb-2 h-12 flex items-center justify-center px-2">
          <p id="abstract-progress-text" class="text-sm text-gray-700 dark:text-gray-300 text-center leading-tight">
            Starting evaluation...
          </p>
        </div>

        <!-- Progress Percentage -->
        <p id="abstract-progress-percentage" class="text-xs text-gray-500 dark:text-gray-500">
          0%
        </p>

        <!-- Cancel Button -->
        <button id="cancel-abstract-evaluation"
                class="mt-4 px-4 py-2 text-sm bg-gray-500 hover:bg-gray-600 text-white
                       rounded-lg transition-colors">
          Cancel
        </button>
      </div>
    </div>
  `;

  document.body.appendChild(modal);
}

/**
 * Update progress in the abstract evaluation modal.
 */
export function updateAbstractEvaluationProgress(
  message: string,
  progress: number,
): void {
  const progressBar = document.getElementById("abstract-progress-bar");
  const progressText = document.getElementById("abstract-progress-text");
  const progressPercentage = document.getElementById("abstract-progress-percentage");

  if (progressBar) {
    progressBar.style.width = `${progress}%`;
  }

  if (progressText) {
    progressText.textContent = message;
  }

  if (progressPercentage) {
    progressPercentage.textContent = `${Math.round(progress)}%`;
  }
}

/**
 * Hide the abstract evaluation modal.
 */
export function hideAbstractEvaluationModal(): void {
  const modal = document.getElementById("abstract-evaluation-modal");
  if (modal) {
    modal.remove();
  }
}

/**
 * Show error message in abstract tab.
 */
export function showAbstractError(message: string, isError = true): void {
  const errorEl = document.getElementById("abstract-error");
  const errorMessage = document.getElementById("abstract-error-message");

  if (errorEl && errorMessage) {
    errorMessage.textContent = message;
    if (isError) {
      // Amber for temporary/retryable errors and validation issues
      errorEl.className =
        "rounded-lg border border-amber-500 bg-amber-100/50 p-4 " +
        "text-amber-700 dark:bg-amber-900/20 dark:text-amber-300";
    } else {
      errorEl.className =
        "rounded-lg border border-green-500 bg-green-100/50 p-4 " +
        "text-green-700 dark:bg-green-900/20 dark:text-green-300";
    }
    errorEl.classList.remove("hidden");
  }
}

/**
 * Show completion notification for abstract evaluation.
 */
export function showAbstractCompletionNotification(title: string): void {
  const notification = document.createElement("div");
  notification.className =
    "fixed top-4 right-4 z-50 rounded-lg bg-green-600 p-4 text-white shadow-lg dark:bg-green-700";

  notification.innerHTML = `
    <div class="flex items-center gap-3">
      <svg class="h-5 w-5 text-green-200" fill="currentColor" viewBox="0 0 20 20">
        <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd" />
      </svg>
      <div>
        <p class="font-medium">Abstract Evaluation Complete!</p>
        <p class="text-sm text-green-200 line-clamp-1">
          ${title.length > 50 ? title.substring(0, 50) + "..." : title}
        </p>
      </div>
    </div>
  `;

  document.body.appendChild(notification);

  setTimeout(() => {
    if (document.body.contains(notification)) {
      notification.remove();
    }
  }, 4000);
}

/**
 * Cache an abstract evaluation result.
 */
export function cacheAbstractEvaluation(result: AbstractEvaluationResponse): void {
  localStorage.setItem(
    `abstract-evaluation-cache-${result.id}`,
    JSON.stringify(result),
  );

  const currentList = localStorage.getItem("abstract-evaluations-list");
  const evaluationIds =
    parseStoredJson(currentList, z.array(z.string()), "abstract-evaluations-list") ??
    [];
  if (!evaluationIds.includes(result.id)) {
    evaluationIds.push(result.id);
    localStorage.setItem("abstract-evaluations-list", JSON.stringify(evaluationIds));
  }
}
