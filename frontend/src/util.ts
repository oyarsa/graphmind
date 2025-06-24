/**
 * Sleep a given number of milliseconds.
 *
 * @param ms Number of milliseconds to sleep.
 */
export async function sleep(ms: number): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Assert that a condition is true, throwing an error if it's not.
 * This function acts as a type guard, narrowing types when the assertion passes.
 *
 * @param condition The condition to check.
 * @param message Optional error message (defaults to "Assertion failed").
 * @throws {Error} If the condition is false.
 */
export function assert(condition: unknown, message?: string): asserts condition {
  if (!condition) {
    throw new Error(message ?? "Assertion failed");
  }
}

/**
 * Get an element by ID and assert that it exists.
 * This utility reduces boilerplate by combining getElementById with existence checking.
 *
 * @param id The element ID to search for.
 * @returns The found HTML element.
 * @throws {Error} If the element is not found.
 */
export function requireElement(id: string): HTMLElement {
  const element = document.getElementById(id);
  assert(element, `Element with ID '${id}' not found`);
  return element;
}

/**
 * Show a styled dialog with a message.
 * This replaces the browser's alert() function with a custom dialog.
 *
 * @param message The message to display in the dialog.
 * @param title Optional title for the dialog (defaults to "Alert").
 * @returns A promise that resolves when the dialog is closed.
 */
export function showDialog(message: string, title: string): Promise<void> {
  return new Promise((resolve) => {
    const overlay = requireElement("dialogOverlay");
    const titleElement = requireElement("dialogTitle");
    const messageElement = requireElement("dialogMessage");
    const okButton = requireElement("dialogOk");

    titleElement.textContent = title;
    messageElement.textContent = message;

    const closeDialog = () => {
      overlay.style.display = "none";
      okButton.removeEventListener("click", closeDialog);
      document.removeEventListener("keydown", handleKeydown);
      resolve();
    };

    const handleKeydown = (event: KeyboardEvent) => {
      if (event.key === "Escape" || event.key === "Enter") {
        event.preventDefault();
        closeDialog();
      }
    };

    okButton.addEventListener("click", closeDialog);
    document.addEventListener("keydown", handleKeydown);

    overlay.style.display = "block";
    okButton.focus();
  });
}

/**
 * Show error page with message and retry button.
 *
 * @param error The error that occurred.
 */
export function showInitError(error: Error): void {
  document.body.innerHTML = `
    <div class="flex min-h-screen items-center justify-center bg-black p-8">
      <div class="max-w-md text-center rounded-lg border border-red-500 bg-red-900/20 p-8">
        <h1 class="mb-4 text-3xl font-bold text-red-400">Failed to load application</h1>
        <p class="mb-6 text-lg text-red-300">${error.message}</p>
        <button
          onclick="location.reload()"
          class="rounded-lg bg-gradient-to-br from-red-500 to-red-700 px-6 py-3 text-sm
                 font-semibold text-white transition-all duration-200
                 hover:-translate-y-0.5 hover:from-red-400 hover:to-red-600"
        >
          Retry
        </button>
      </div>
    </div>
  `;
}

/**
 * Retry a function with exponential backoff.
 *
 * @param fn The function to retry.
 * @param maxRetries Maximum number of retry attempts (default: 3).
 * @param baseDelay Base delay in milliseconds (default: 1000).
 * @returns Promise that resolves with the function result or rejects with the final
 *          error.
 */
export async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  maxRetries = 3,
  baseDelay = 1000,
): Promise<T> {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      console.warn(`Attempt ${attempt}/${maxRetries} failed:`, error);

      if (attempt === maxRetries) {
        console.error("All attempts failed");
        throw error;
      }

      await sleep(baseDelay * attempt);
    }
  }

  // This should never be reached due to the throw above, but TypeScript requires it
  throw new Error("Unexpected retry logic error");
}

/**
 * Wait for DOM to be ready.
 *
 * @returns Promise that resolves when DOM is ready.
 */
export async function waitForDOM(): Promise<void> {
  if (document.readyState === "loading") {
    await new Promise((resolve) => {
      document.addEventListener("DOMContentLoaded", resolve);
    });
  }
}

/**
 * Check if device is mobile.
 *
 * @returns True if the device is mobile.
 */
export function isMobileDevice(): boolean {
  return (
    /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
      navigator.userAgent,
    ) || window.innerWidth < 768
  );
}

/**
 * Show mobile-only message.
 */
export function showMobileMessage(): void {
  document.body.innerHTML = `
    <div class="flex min-h-screen items-center justify-center bg-black p-8">
      <div class="max-w-md text-center">
        <h1 class="mb-4 text-3xl font-bold text-blue-500">Desktop Only</h1>
        <p class="text-lg text-gray-300">
          This application is only supported on desktop browsers.
        </p>
        <p class="mt-4 text-sm text-gray-500">
          Please visit this site on a desktop or laptop computer for the best experience.
        </p>
      </div>
    </div>
  `;
}

/**
 * Check if a response is ok, throwing an error with the provided message if not.
 *
 * @param response The fetch response to check.
 * @param message The error message to include in the thrown error.
 * @throws {Error} If the response is not ok.
 */
export function assertResponse(response: Response, message: string): void {
  if (!response.ok) {
    throw new Error(`${message}: ${response.status} ${response.statusText}`);
  }
}
