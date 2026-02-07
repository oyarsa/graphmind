/**
 * Cache management for evaluated papers.
 */

import {
  GraphResult,
  GraphResultMulti,
  GraphResultSchema,
  GraphResultMultiSchema,
  PaperSearchResults,
} from "./model";
import { parseStoredJson } from "./storage";

/** Extended interface for cached papers */
export interface CachedPaperSearchItem {
  title: string;
  abstract: string;
  authors: string[];
  year?: number;
  arxiv_id?: string;
  _isCached?: boolean;
  _cachedPaperId?: string;
}

export type ArxivMode = "default" | "multi";

/**
 * Get all cached papers from localStorage for the specified mode.
 */
export function getCachedPapers(mode: ArxivMode): (GraphResult | GraphResultMulti)[] {
  if (mode === "multi") {
    return getCachedPapersMulti();
  } else {
    return getCachedPapersDefault();
  }
}

/**
 * Get all cached default papers from localStorage.
 */
function getCachedPapersDefault(): GraphResult[] {
  const cacheKeys = Object.keys(localStorage).filter(
    key => key.startsWith("paper-cache-") && !key.includes("paper-cache-multi-"),
  );

  const cachedPapers: GraphResult[] = [];

  for (const key of cacheKeys) {
    const cachedData = localStorage.getItem(key);
    const graphResult = parseStoredJson(cachedData, GraphResultSchema, key);
    if (graphResult) {
      cachedPapers.push(graphResult);
    }
  }

  return cachedPapers.sort((a, b) => a.paper.title.localeCompare(b.paper.title));
}

/**
 * Get all cached multi-perspective papers from localStorage.
 */
function getCachedPapersMulti(): GraphResultMulti[] {
  const cacheKeys = Object.keys(localStorage).filter(key =>
    key.startsWith("paper-cache-multi-"),
  );

  const cachedPapers: GraphResultMulti[] = [];

  for (const key of cacheKeys) {
    const cachedData = localStorage.getItem(key);
    const graphResult = parseStoredJson(cachedData, GraphResultMultiSchema, key);
    if (graphResult) {
      cachedPapers.push(graphResult);
    }
  }

  return cachedPapers.sort((a, b) => a.paper.title.localeCompare(b.paper.title));
}

/**
 * Convert cached GraphResult objects to PaperSearchResults format.
 */
export function convertCachedPapersToSearchResults(
  cachedPapers: (GraphResult | GraphResultMulti)[],
): Omit<PaperSearchResults, "items"> & { items: CachedPaperSearchItem[] } {
  const items: CachedPaperSearchItem[] = cachedPapers.map(graphResult => {
    const paper = graphResult.paper;
    return {
      title: paper.title,
      abstract: paper.abstract,
      authors: paper.authors,
      year: paper.year,
      ...(paper.arxiv_id && { arxiv_id: paper.arxiv_id }),
      _isCached: true,
      _cachedPaperId: paper.id,
    };
  });

  return {
    query: "cached",
    total: items.length,
    items: items,
  };
}

/**
 * Get cache keys for the specified mode.
 */
export function getCacheKeys(mode: ArxivMode): string[] {
  return Object.keys(localStorage).filter(key => {
    if (mode === "multi") {
      return key.startsWith("paper-cache-multi-");
    } else {
      return key.startsWith("paper-cache-") && !key.includes("paper-cache-multi-");
    }
  });
}

/**
 * Clear all cached papers for the specified mode.
 */
export function clearCache(mode: ArxivMode): void {
  const cacheKeys = getCacheKeys(mode);
  cacheKeys.forEach(key => localStorage.removeItem(key));
}

/**
 * Find a cached paper by arXiv ID.
 */
export function findCachedPaperByArxivId(
  arxivId: string,
  mode: ArxivMode,
): GraphResult | GraphResultMulti | null {
  const cacheKeys = getCacheKeys(mode);
  const modeText = mode === "multi" ? "multi-perspective" : "default";
  console.log(
    `[Cache] Found ${cacheKeys.length} cached ${modeText} papers in localStorage`,
  );

  for (const key of cacheKeys) {
    const cachedData = localStorage.getItem(key);
    const graphResult =
      mode === "multi"
        ? parseStoredJson(cachedData, GraphResultMultiSchema, key)
        : parseStoredJson(cachedData, GraphResultSchema, key);

    if (!graphResult) {
      continue;
    }

    console.log(`[Cache] Checking ${key}: arxiv_id=${graphResult.paper.arxiv_id}`);
    if (graphResult.paper.arxiv_id === arxivId) {
      console.log(
        `[Cache] HIT! Found ${modeText} paper ${arxivId} in cache with key ${key}`,
      );
      return graphResult;
    }
  }

  console.log(`[Cache] MISS! Paper ${arxivId} not found in cache`);
  return null;
}

/**
 * Store a paper result in the cache.
 */
export function storePaperInCache(
  paperId: string,
  result: GraphResult | GraphResultMulti,
  mode: ArxivMode,
): void {
  const cacheKey =
    mode === "multi" ? `paper-cache-multi-${paperId}` : `paper-cache-${paperId}`;
  console.log(`[Cache] Storing ${mode} paper in cache with key ${cacheKey}`);
  localStorage.setItem(cacheKey, JSON.stringify(result));
}
