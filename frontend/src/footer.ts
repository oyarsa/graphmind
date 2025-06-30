const sep = "•";
const github = `<a href="https://github.com/italo/paper-hypergraph" target="_blank" rel="noopener noreferrer" class="hover:text-gray-600 dark:hover:text-gray-400 underline">GitHub</a>`;
const license = "AGPL-3.0-or-later";

/**
 * Creates and returns a footer element with project version and build information
 */
export function createFooter(): HTMLElement {
  const footer = document.createElement("footer");
  footer.className =
    "mt-auto py-2 text-center text-xs text-gray-500 dark:text-gray-600";

  footer.innerHTML = `v${VERSION} ${sep} ${BUILD_TIME} ${sep} ${github} ${sep} ${license}`;

  return footer;
}

/**
 * Adds a footer to the document body
 */
export function addFooter(): void {
  document.body.appendChild(createFooter());
}
