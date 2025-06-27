const sep = "•";
/**
 * Creates and returns a footer element with project version and build information
 */
export function createFooter(): HTMLElement {
  const footer = document.createElement("footer");
  footer.className =
    "mt-auto py-2 text-center text-xs text-gray-500 dark:text-gray-600";

  footer.textContent = `v${VERSION} ${sep} ${BUILD_TIME} ${sep} GPL-3.0-or-later`;

  return footer;
}

/**
 * Adds a footer to the document body
 */
export function addFooter(): void {
  const footer = createFooter();
  document.body.appendChild(footer);
}
