/**
 * Build application-relative routes that respect Vite's BASE_URL.
 */
export function buildAppPath(path: string): string {
  const baseUrl = import.meta.env.BASE_URL;
  const normalizedBase = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
  const normalizedPath = path.replace(/^\/+/, "");
  return `${normalizedBase}${normalizedPath}`;
}

/**
 * Build a page URL with optional query parameters.
 */
export function buildPageUrl(page: string, query?: Record<string, string>): string {
  const pagePath = buildAppPath(`pages/${page}`);
  if (!query) {
    return pagePath;
  }

  const params = new URLSearchParams(query);
  const queryString = params.toString();
  return queryString.length > 0 ? `${pagePath}?${queryString}` : pagePath;
}
