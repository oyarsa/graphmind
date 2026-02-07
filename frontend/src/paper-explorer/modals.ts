/**
 * Modal management utilities for the Paper Explorer.
 */

import { renderLatex } from "./helpers";
import { escapeHtml } from "../util";
import {
  DEFAULT_SETTINGS,
  applySettingsToForm,
  getSettingsFromForm,
  loadSettings,
  saveSettings,
} from "./settings";

/**
 * Show the evaluation progress modal.
 */
export function showEvaluationModal(title: string): void {
  hideEvaluationModal();

  const modal = document.createElement("div");
  modal.id = "evaluation-modal";
  modal.className =
    "fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm";

  modal.innerHTML = `
    <div class="mx-4 w-96 rounded-lg bg-white p-6 shadow-xl dark:bg-gray-800">
      <div class="text-center">
        <h3 class="mb-2 text-lg font-semibold text-gray-900 dark:text-white">
          Evaluating Paper
        </h3>
        <p class="mb-4 text-sm text-gray-600 dark:text-gray-400 line-clamp-2">
          ${renderLatex(title)}
        </p>

        <!-- Progress Bar -->
        <div class="mb-4 w-full bg-gray-200 rounded-full h-2 dark:bg-gray-700">
          <div id="progress-bar" class="bg-teal-600 h-2 rounded-full transition-all duration-300"
               style="width: 0%">
          </div>
        </div>

        <!-- Progress Text -->
        <div class="mb-2 h-12 flex items-center justify-center px-2">
          <p id="progress-text" class="text-sm text-gray-700 dark:text-gray-300 text-center leading-tight">
            Starting evaluation...
          </p>
        </div>

        <!-- Progress Percentage -->
        <p id="progress-percentage" class="text-xs text-gray-500 dark:text-gray-500">
          0%
        </p>

        <!-- Cancel Button -->
        <button id="cancel-arxiv-evaluation"
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
 * Update the evaluation progress display.
 */
export function updateEvaluationProgress(message: string, progress: number): void {
  const progressBar = document.getElementById("progress-bar");
  const progressText = document.getElementById("progress-text");
  const progressPercentage = document.getElementById("progress-percentage");

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
 * Show an error in the evaluation modal.
 */
export function showEvaluationError(
  errorMessage: string,
  onRetry: () => void,
  onDismiss: () => void,
): void {
  const modal = document.getElementById("evaluation-modal");
  if (!modal) return;

  const modalContent = modal.querySelector("div");
  if (!modalContent) return;

  modalContent.innerHTML = `
    <div class="text-center">
      <div class="mb-4">
        <div class="mx-auto h-12 w-12 rounded-full bg-amber-100 dark:bg-amber-900 flex
                    items-center justify-center">
          <svg class="h-6 w-6 text-amber-600 dark:text-amber-400" fill="none"
               viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.962-.833-2.732 0L4.082 18.5c-.77.833.192 2.5 1.732 2.5z"
            />
          </svg>
        </div>
      </div>
      <h3 class="mb-2 text-lg font-semibold text-gray-900 dark:text-white">
        Evaluation Failed
      </h3>
      <p class="mb-6 text-sm text-gray-600 dark:text-gray-400">
        ${escapeHtml(errorMessage)}
      </p>

      <div class="flex gap-3 justify-center">
        <button id="retry-evaluation"
                class="px-4 py-2 text-sm bg-teal-600 hover:bg-teal-700 text-white
                       rounded-lg transition-colors">
          Try Again
        </button>
        <button id="dismiss-error"
                class="px-4 py-2 text-sm bg-gray-500 hover:bg-gray-600 text-white
                       rounded-lg transition-colors">
          Dismiss
        </button>
      </div>
    </div>
  `;

  const retryButton = document.getElementById("retry-evaluation");
  const dismissButton = document.getElementById("dismiss-error");

  if (retryButton) {
    retryButton.addEventListener("click", onRetry);
  }

  if (dismissButton) {
    dismissButton.addEventListener("click", onDismiss);
  }
}

/**
 * Hide the evaluation modal.
 */
export function hideEvaluationModal(): void {
  const modal = document.getElementById("evaluation-modal");
  if (modal) {
    modal.remove();
  }
}

/**
 * Show the evaluation settings modal for a paper.
 */
export function showEvaluationSettingsModal(title: string): void {
  const modal = document.getElementById("evaluation-settings-modal");
  const paperTitleEl = document.getElementById("evaluation-paper-title");

  if (!modal || !paperTitleEl) {
    console.error("Evaluation settings modal elements not found");
    return;
  }

  paperTitleEl.textContent = title;
  modal.classList.remove("hidden");
  modal.classList.add("flex");
}

/**
 * Hide the evaluation settings modal.
 */
export function hideEvaluationSettingsModal(): void {
  const modal = document.getElementById("evaluation-settings-modal");
  if (modal) {
    modal.classList.add("hidden");
    modal.classList.remove("flex");
  }
}

/**
 * Setup the settings modal functionality.
 */
export function setupSettingsModal(): void {
  const settingsButton = document.getElementById("settings-button");
  const settingsModal = document.getElementById("settings-modal");
  const closeButton = document.getElementById("settings-modal-close");
  const saveButton = document.getElementById("settings-save");
  const resetButton = document.getElementById("settings-reset");

  if (
    !settingsButton ||
    !settingsModal ||
    !closeButton ||
    !saveButton ||
    !resetButton
  ) {
    console.warn("Settings modal elements not found");
    return;
  }

  const hideSettingsModal = () => {
    settingsModal.classList.add("hidden");
    settingsModal.classList.remove("flex");
  };

  settingsButton.addEventListener("click", () => {
    applySettingsToForm(loadSettings());
    settingsModal.classList.remove("hidden");
    settingsModal.classList.add("flex");
  });

  closeButton.addEventListener("click", hideSettingsModal);

  settingsModal.addEventListener("click", e => {
    if (e.target === settingsModal) {
      hideSettingsModal();
    }
  });

  document.addEventListener("keydown", e => {
    if (e.key === "Escape" && !settingsModal.classList.contains("hidden")) {
      hideSettingsModal();
    }
  });

  saveButton.addEventListener("click", () => {
    const settings = getSettingsFromForm();
    saveSettings(settings);
    hideSettingsModal();
  });

  resetButton.addEventListener("click", () => {
    applySettingsToForm(DEFAULT_SETTINGS);
  });
}

/**
 * Setup the help modal functionality.
 */
export function setupHelpModal(): void {
  const helpButton = document.getElementById("help-button");
  const helpModal = document.getElementById("help-modal");
  const helpModalClose = document.getElementById("help-modal-close");

  if (!helpButton || !helpModal || !helpModalClose) {
    console.warn("Help modal elements not found");
    return;
  }

  const hideHelpModal = () => {
    helpModal.classList.add("hidden");
    helpModal.classList.remove("flex");
    localStorage.setItem("paper-explorer-help-seen", "true");
  };

  const hasSeenHelp = localStorage.getItem("paper-explorer-help-seen");
  if (!hasSeenHelp) {
    helpModal.classList.remove("hidden");
    helpModal.classList.add("flex");
  }

  helpButton.addEventListener("click", () => {
    helpModal.classList.remove("hidden");
    helpModal.classList.add("flex");
  });

  helpModalClose.addEventListener("click", hideHelpModal);

  helpModal.addEventListener("click", e => {
    if (e.target === helpModal) {
      hideHelpModal();
    }
  });

  document.addEventListener("keydown", e => {
    if (e.key === "Escape" && !helpModal.classList.contains("hidden")) {
      hideHelpModal();
    }
  });
}

/**
 * Request notification permission from the browser.
 */
export async function requestNotificationPermission(): Promise<void> {
  if (!("Notification" in window)) {
    console.log("This browser does not support notifications");
    return;
  }

  if (Notification.permission === "default") {
    try {
      await Notification.requestPermission();
    } catch (error) {
      console.warn("Failed to request notification permission:", error);
    }
  }
}

/**
 * Show a browser notification when evaluation completes.
 */
export function showCompletionNotification(paperTitle: string): void {
  if (!("Notification" in window) || Notification.permission !== "granted") {
    return;
  }

  try {
    const titleTrunc = `${paperTitle.substring(0, 100)}${paperTitle.length > 100 ? "..." : ""}`;
    const notification = new Notification("Paper Evaluation Complete", {
      body: `Finished evaluating: ${titleTrunc}`,
      icon: "/favicon.ico",
      tag: "paper-evaluation-complete",
      requireInteraction: false,
      silent: false,
    });

    setTimeout(() => notification.close(), 5000);

    notification.onclick = () => {
      window.focus();
      notification.close();
    };
  } catch (error) {
    console.warn("Failed to show notification:", error);
  }
}
