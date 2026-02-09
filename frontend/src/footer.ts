const link = (link: string, txt: string): string => `
  <a href="${link}"
     rel="noopener noreferrer"
     class="hover:text-gray-600 dark:hover:text-gray-400 underline"
     >${txt}</a>
`;
const sep = "•";
const api = link(API_DOCS_URL, "API docs");
const github = link("https://github.com/oyarsa/graphmind", "Source Code");
const license = "AGPL-3.0-or-later";

/**
 * Creates and returns a footer element with project version and build information
 */
export function createFooter(): HTMLElement {
  const footer = document.createElement("footer");
  footer.className =
    "mt-auto py-2 text-center text-xs text-gray-500 dark:text-gray-600";

  footer.innerHTML = `v${VERSION} ${sep} ${api} ${sep} ${github} ${sep} ${license}`;

  return footer;
}

/**
 * Adds a footer to the document body
 */
export function addFooter(): void {
  document.body.appendChild(createFooter());
}
