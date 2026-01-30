/**
 * End-to-end tests for the paper evaluation frontend.
 *
 * These tests use Playwright to verify the full evaluation flow:
 * - Search functionality
 * - Paper evaluation via API
 * - Detail page rendering with correct data
 *
 * Tests are grouped by paper to minimize API calls - each paper is evaluated
 * only once and subsequent tests reuse the cached evaluation via shared
 * browser context.
 *
 * Run with: just e2e (requires backend on :8000 and frontend on :5173)
 */

import { test as base, expect, Page, BrowserContext } from "@playwright/test";

// Paper definitions (arXiv ID, title, expected year)
const ATTENTION_PAPER = {
  arxivId: "1706.03762",
  title: "Attention Is All You Need",
  expectedYear: 2017,
};

const SLEEPER_AGENTS_PAPER = {
  arxivId: "2401.05566",
  title: "Sleeper Agents: Training Deceptive LLMs",
  expectedYear: 2024,
};

// =============================================================================
// Custom test fixture that shares browser context within serial test groups
// =============================================================================

// Worker-scoped fixtures persist across all tests in the same worker
interface WorkerFixtures {
  sharedContext: BrowserContext;
  sharedPage: Page;
}

const test = base.extend<object, WorkerFixtures>({
  // Create a shared browser context that persists across tests in the same worker
  sharedContext: [
    async ({ browser }, use) => {
      const context = await browser.newContext();
      await use(context);
      await context.close();
    },
    { scope: "worker" },
  ],

  // Create a page from the shared context - also worker-scoped
  sharedPage: [
    async ({ sharedContext }, use) => {
      const page = await sharedContext.newPage();
      await use(page);
      // Don't close the page - let context closure handle it
    },
    { scope: "worker" },
  ],
});

// =============================================================================
// Helper classes and functions
// =============================================================================

interface CheckResult {
  name: string;
  passed: boolean;
  message: string;
}

/**
 * Collects checks on a paper evaluation and reports all failures at the end.
 */
class EvaluationChecker {
  private checks: CheckResult[] = [];

  constructor(private paperTitle: string) {}

  check(name: string, condition: boolean, message = ""): void {
    this.checks.push({
      name,
      passed: condition,
      message: condition ? "" : message,
    });
  }

  checkEqual<T>(name: string, actual: T, expected: T): void {
    const passed = actual === expected;
    this.checks.push({
      name,
      passed,
      message: passed ? "" : `expected ${expected}, got ${actual}`,
    });
  }

  checkInRange(name: string, value: number, min: number, max: number): void {
    const passed = value >= min && value <= max;
    this.checks.push({
      name,
      passed,
      message: passed ? "" : `expected ${min}-${max}, got ${value}`,
    });
  }

  checkNotEmpty(name: string, value: unknown): void {
    const passed = value !== null && value !== undefined && value !== "";
    this.checks.push({
      name,
      passed,
      message: passed ? "" : "value is empty or undefined",
    });
  }

  checkMaxLength(name: string, length: number, maxLen: number): void {
    const passed = length <= maxLen;
    this.checks.push({
      name,
      passed,
      message: passed ? "" : `expected at most ${maxLen}, got ${length}`,
    });
  }

  checkNoLatex(name: string, text: string): void {
    const latexPatterns = [
      /\\cite[pt]?\{/,
      /\\footnote\{/,
      /\\textbf\{/,
      /\\emph\{/,
      /~\\/,
    ];
    const found = latexPatterns.filter(p => p.test(text)).map(p => p.source);
    const passed = found.length === 0;
    this.checks.push({
      name,
      passed,
      message: passed ? "" : `found LaTeX patterns: ${found.join(", ")}`,
    });
  }

  checkNoBadSummaryPatterns(name: string, text: string): void {
    const badPatterns = [
      /^The [Rr]elated [Pp]aper/,
      /^The [Mm]ain [Pp]aper/,
      /^This related paper/,
    ];
    const found = badPatterns.filter(p => p.test(text)).map(p => p.source);
    const passed = found.length === 0;
    this.checks.push({
      name,
      passed,
      message: passed
        ? ""
        : `text starts with bad pattern. First 50 chars: ${text.slice(0, 50)}...`,
    });
  }

  checkYearNotAfter(name: string, year: number | null, maxYear: number): void {
    if (year === null) {
      this.checks.push({ name, passed: true, message: "" });
      return;
    }
    const passed = year <= maxYear;
    this.checks.push({
      name,
      passed,
      message: passed ? "" : `expected year <= ${maxYear}, got ${year}`,
    });
  }

  assertAllPassed(): void {
    const failures = this.checks.filter(c => !c.passed);
    if (failures.length > 0) {
      const failureMessages = failures
        .map(f => `  - ${f.name}: ${f.message}`)
        .join("\n");
      throw new Error(
        `Paper "${this.paperTitle}": ${failures.length} check(s) failed:\n${failureMessages}`,
      );
    }
  }
}

/**
 * Clear localStorage to start fresh.
 */
async function clearLocalStorage(page: Page): Promise<void> {
  await page.evaluate(() => {
    const keys = Object.keys(localStorage);
    keys.forEach(key => {
      if (key.startsWith("paper-cache-") || key.startsWith("abstract-evaluation-")) {
        localStorage.removeItem(key);
      }
    });
    localStorage.removeItem("abstract-evaluations-list");
    localStorage.removeItem("paper-explorer-help-seen");
  });
}

/**
 * Close help modal if shown.
 */
async function closeHelpModal(page: Page): Promise<void> {
  const helpModal = page.locator("#help-modal.flex");
  if (await helpModal.isVisible()) {
    await page.locator("#help-modal-close").click();
    await page.waitForTimeout(300);
  }
}

/**
 * Navigate to search page and optionally clear cache.
 */
async function goToSearchPage(
  page: Page,
  options: { clearCache?: boolean } = {},
): Promise<void> {
  await page.goto("pages/search.html");
  await page.waitForLoadState("networkidle");

  if (options.clearCache) {
    await clearLocalStorage(page);
  }

  await closeHelpModal(page);
}

/**
 * Search for a paper by title in the arXiv tab with retry logic for rate limiting.
 */
async function searchPaper(
  page: Page,
  title: string,
  options: { maxRetries?: number; retryDelayMs?: number } = {},
): Promise<void> {
  const maxRetries = options.maxRetries ?? 5;
  const retryDelayMs = options.retryDelayMs ?? 5000; // 5 seconds for backend rate limit

  // Ensure we're on the arXiv tab
  const arxivTab = page.locator("#arxiv-tab");
  if (!(await arxivTab.getAttribute("class"))?.includes("active")) {
    await arxivTab.click();
    await page.waitForTimeout(500);
  }

  let lastError: string | null = null;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    // Clear any previous search
    const searchInput = page.locator("#arxiv-search-input");
    await searchInput.fill(title);
    await page.locator("#arxiv-search-button").click();

    // Wait for either results or error
    try {
      // Wait for results container to become visible
      await page.waitForSelector("#arxiv-papers-container:not(.hidden)", {
        timeout: 60000, // 60 seconds
      });
      return; // Success
    } catch {
      // Check for error message
      const errorEl = page.locator("#arxiv-error-message");
      const errorText = (await errorEl.isVisible())
        ? await errorEl.textContent()
        : null;
      lastError = errorText ?? "timeout";

      // Retry on any failure, not just rate limiting
      if (attempt < maxRetries) {
        const jitter = Math.random() * 2000; // 0-2 second random jitter
        const waitTime = retryDelayMs + jitter;
        console.log(
          `Search attempt ${attempt + 1}/${maxRetries + 1} failed (${lastError}), waiting ${(waitTime / 1000).toFixed(1)}s...`,
        );
        await page.waitForTimeout(waitTime);
        continue;
      }
    }
  }

  throw new Error(
    `Search failed after ${maxRetries + 1} attempts. Last error: ${lastError?.slice(0, 200)}`,
  );
}

/**
 * Click on a paper to start evaluation.
 */
async function clickPaperCard(page: Page, arxivId: string): Promise<void> {
  // Find the paper card with the correct arXiv link
  const paperCard = page.locator(
    `#arxiv-papers-container > div:has(a[href*="${arxivId}"])`,
  );
  await paperCard.click();
}

/**
 * Wait for and confirm evaluation in the modal.
 */
async function startEvaluation(page: Page): Promise<void> {
  // Wait for evaluation settings modal
  await page.waitForSelector("#evaluation-settings-modal.flex", {
    timeout: 5000,
  });

  // Click start evaluation
  await page.locator("#start-evaluation").click();
}

/**
 * Wait for evaluation to complete and navigate to detail page.
 */
async function waitForEvaluationComplete(page: Page): Promise<void> {
  // Wait for the evaluation modal progress
  await page.waitForSelector("#evaluation-modal", { timeout: 10000 });

  // Wait for navigation to detail page (evaluation complete)
  await page.waitForURL("**/detail.html**", {
    timeout: 6 * 60 * 1000, // 6 minutes for evaluation
  });

  // Wait for content to load
  await page.waitForSelector("#paper-content:not(.hidden)", { timeout: 10000 });
}

/**
 * Full flow: search, click, evaluate, wait for completion.
 */
async function evaluatePaper(
  page: Page,
  paper: { title: string; arxivId: string },
): Promise<void> {
  await searchPaper(page, paper.title);
  await clickPaperCard(page, paper.arxivId);
  await startEvaluation(page);
  await waitForEvaluationComplete(page);
}

/**
 * Navigate to a cached paper's detail page (no evaluation modal).
 */
async function goToCachedPaperDetail(
  page: Page,
  paper: { title: string; arxivId: string },
): Promise<void> {
  await goToSearchPage(page);
  await searchPaper(page, paper.title);
  await clickPaperCard(page, paper.arxivId);

  // Should navigate directly to detail (cached) - no modal
  await page.waitForURL("**/detail.html**", { timeout: 10000 });
  await page.waitForSelector("#paper-content:not(.hidden)", { timeout: 10000 });
}

/**
 * Extract data from the detail page for verification.
 */
async function extractDetailPageData(page: Page): Promise<{
  title: string;
  year: number | null;
  noveltyScore: string | null;
  hasPaperSummary: boolean;
  hasConclusion: boolean;
  supportingEvidenceCount: number;
  contradictoryEvidenceCount: number;
  relatedPapersCount: number;
  hasGraph: boolean;
  evidenceTexts: string[];
  relatedPaperYears: (number | null)[];
}> {
  return await page.evaluate(() => {
    const getTextContent = (selector: string): string | null => {
      const el = document.querySelector(selector);
      return el?.textContent?.trim() ?? null;
    };

    // Extract novelty score
    const noveltyEl = document.getElementById("paper-rating");
    const noveltyScore = noveltyEl?.textContent?.trim() ?? null;

    // Extract year
    const yearEl = document.getElementById("paper-year");
    const yearText = yearEl?.textContent?.trim();
    const year = yearText ? parseInt(yearText, 10) : null;

    // Check for paper summary (in structured evaluation)
    const structuredEval = document.getElementById("structured-evaluation");
    const hasPaperSummary =
      structuredEval?.innerHTML.includes("Paper Summary") ?? false;
    const hasConclusion = structuredEval?.innerHTML.includes("Result") ?? false;

    // Count evidence items
    const supportingSection = structuredEval?.querySelector(
      '.rounded-lg:has([class*="bg-green-500"])',
    );
    const contradictorySection = structuredEval?.querySelector(
      '.rounded-lg:has([class*="bg-red-500"])',
    );

    const supportingEvidenceCount =
      supportingSection?.querySelectorAll("li").length ?? 0;
    const contradictoryEvidenceCount =
      contradictorySection?.querySelectorAll("li").length ?? 0;

    // Count related papers
    const relatedPapersContent = document.getElementById("related-papers-content");
    const relatedPapersCount =
      relatedPapersContent?.querySelectorAll('[id^="related-card-"]').length ?? 0;

    // Check if graph exists
    const graphSvg = document.getElementById("graph-svg");
    const hasGraph = graphSvg ? graphSvg.innerHTML.trim().length > 0 : false;

    // Extract evidence texts for LaTeX checks
    const evidenceTexts: string[] = [];
    structuredEval?.querySelectorAll("li span.text-sm").forEach(el => {
      const text = el.textContent?.trim();
      if (text) evidenceTexts.push(text);
    });

    // Extract related paper years (from expanded cards if any, otherwise from cards)
    const relatedPaperYears: (number | null)[] = [];
    relatedPapersContent?.querySelectorAll('[id^="related-card-"]').forEach(card => {
      // Try to find year in the card metadata
      const authorYearDiv = card.querySelector(
        '[id^="expanded-content-"] .text-gray-600',
      );
      if (authorYearDiv) {
        const match = authorYearDiv.textContent?.match(/\((\d{4})\)/);
        if (match) {
          relatedPaperYears.push(parseInt(match[1], 10));
        }
      }
    });

    return {
      title: getTextContent("#paper-title") ?? "",
      year,
      noveltyScore,
      hasPaperSummary,
      hasConclusion,
      supportingEvidenceCount,
      contradictoryEvidenceCount,
      relatedPapersCount,
      hasGraph,
      evidenceTexts,
      relatedPaperYears,
    };
  });
}

/**
 * Extract cached paper data from localStorage for deeper checks.
 */
interface CachedPaperData {
  year: number | null;
  structuredEvaluation: {
    label: number;
    paper_summary: string;
    conclusion: string;
    supporting_evidence: { text: string; source?: string; paper_title?: string }[];
    contradictory_evidence: { text: string; source?: string; paper_title?: string }[];
  } | null;
  relatedPapers: {
    title: string;
    year: number | null;
    source: string;
    polarity: string;
  }[];
}

async function extractCachedPaperData(
  page: Page,
  paperId: string,
): Promise<CachedPaperData | null> {
  return await page.evaluate(id => {
    // Find the cache key for this paper
    const keys = Object.keys(localStorage);
    const cacheKey = keys.find(
      k =>
        k.startsWith("paper-cache-") &&
        !k.includes("multi") &&
        localStorage.getItem(k)?.includes(id),
    );

    if (!cacheKey) return null;

    try {
      // Data from localStorage is untyped - we validate it at runtime
      // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
      const data = JSON.parse(localStorage.getItem(cacheKey) ?? "{}");
      // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-member-access
      const paper = data.paper;
      // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-member-access
      const related = data.related ?? [];

      return {
        // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
        year: (paper?.year as number | null) ?? null,
        // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-assignment
        structuredEvaluation: paper?.structured_evaluation ?? null,

        relatedPapers: (
          related as {
            title: string;
            year: number | null;
            source: string;
            polarity: string;
          }[]
        ).map(r => ({
          title: r.title,
          year: r.year,
          source: r.source,
          polarity: r.polarity,
        })),
      };
    } catch {
      return null;
    }
  }, paperId);
}

// =============================================================================
// Test Groups
// =============================================================================

/**
 * Basic tests that don't require paper evaluation.
 */
test.describe("Basic Search Tests", () => {
  test.beforeEach(async ({ page }) => {
    await goToSearchPage(page, { clearCache: true });
  });

  test("search page loads correctly", async ({ page }) => {
    // Check page title
    await expect(page).toHaveTitle(/GraphMind/);

    // Check tabs are visible
    await expect(page.locator("#arxiv-tab")).toBeVisible();
    await expect(page.locator("#abstract-tab")).toBeVisible();
    await expect(page.locator("#json-tab")).toBeVisible();

    // Check search input is visible
    await expect(page.locator("#arxiv-search-input")).toBeVisible();
  });

  test("arXiv search returns results", async ({ page }) => {
    await searchPaper(page, "attention transformer");

    // Check that results are displayed
    const results = page.locator("#arxiv-papers-container > div");
    await expect(results.first()).toBeVisible();

    // Check result count is shown
    const resultCount = page.locator("#arxiv-result-count");
    await expect(resultCount).toContainText("Found");
  });
});

/**
 * Tests for the Attention paper - evaluate once, then run all related tests.
 * Uses sharedPage fixture to share browser context (and localStorage) across tests.
 */
test.describe.serial("Attention Paper Tests", () => {
  test("evaluate and verify detail page", async ({ sharedPage: page }) => {
    // Clear cache and evaluate the paper
    await goToSearchPage(page, { clearCache: true });
    await evaluatePaper(page, ATTENTION_PAPER);

    const checker = new EvaluationChecker(ATTENTION_PAPER.title);

    // Extract page data
    const pageData = await extractDetailPageData(page);

    // Basic structure checks
    checker.checkNotEmpty("title exists", pageData.title);
    checker.checkEqual("year", pageData.year, ATTENTION_PAPER.expectedYear);
    checker.checkNotEmpty("novelty score exists", pageData.noveltyScore);
    checker.check("has paper summary", pageData.hasPaperSummary);
    checker.check("has conclusion", pageData.hasConclusion);
    checker.check("has graph", pageData.hasGraph);

    // Parse novelty score (format: "X/5" or "High"/"Low")
    if (pageData.noveltyScore) {
      const scoreMatch = /^(\d)\/5$/.exec(pageData.noveltyScore);
      if (scoreMatch) {
        const score = parseInt(scoreMatch[1], 10);
        checker.checkInRange("novelty score in range", score, 1, 5);
      }
    }

    // Evidence limits
    checker.checkMaxLength(
      "supporting evidence max 5",
      pageData.supportingEvidenceCount,
      5,
    );
    checker.checkMaxLength(
      "contradictory evidence max 5",
      pageData.contradictoryEvidenceCount,
      5,
    );

    // LaTeX checks on evidence texts
    for (let i = 0; i < Math.min(3, pageData.evidenceTexts.length); i++) {
      checker.checkNoLatex(`evidence[${i}] no LaTeX`, pageData.evidenceTexts[i]);
      checker.checkNoBadSummaryPatterns(
        `evidence[${i}] no bad patterns`,
        pageData.evidenceTexts[i],
      );
    }

    // Extract cached data for deeper checks
    const cachedData = await extractCachedPaperData(page, ATTENTION_PAPER.arxivId);
    if (cachedData) {
      // Check date filtering - related papers should be <= paper year
      for (const related of cachedData.relatedPapers) {
        checker.checkYearNotAfter(
          `related "${related.title.slice(0, 30)}..." year`,
          related.year,
          ATTENTION_PAPER.expectedYear,
        );
      }

      // Check evidence source distribution
      if (cachedData.structuredEvaluation) {
        const supportingSemanticCount =
          cachedData.structuredEvaluation.supporting_evidence.filter(
            e => e.source === "semantic",
          ).length;
        const supportingCitationsCount =
          cachedData.structuredEvaluation.supporting_evidence.filter(
            e => e.source === "citations",
          ).length;

        checker.checkMaxLength("supporting semantic <= 3", supportingSemanticCount, 3);
        checker.checkMaxLength(
          "supporting citations reasonable",
          supportingCitationsCount,
          5,
        );
      }
    }

    checker.assertAllPassed();
  });

  test("detail page sections are collapsible", async ({ sharedPage: page }) => {
    // Navigate to cached paper detail page
    await goToCachedPaperDetail(page, ATTENTION_PAPER);

    // Check that sections exist and can be toggled
    const graphHeader = page.locator("#paper-graph-header");
    const graphContent = page.locator("#paper-graph");

    await expect(graphHeader).toBeVisible();
    await expect(graphContent).toBeVisible();

    // Click to collapse (graph starts expanded)
    await graphHeader.click();
    await page.waitForTimeout(300);
    await expect(graphContent).toBeHidden();

    // Click to expand again
    await graphHeader.click();
    await page.waitForTimeout(300);
    await expect(graphContent).toBeVisible();
  });

  test("cached paper navigates directly to detail", async ({ sharedPage: page }) => {
    // Navigate back to search
    await goToSearchPage(page);

    // Wait for cached papers to appear
    await page.waitForSelector("#arxiv-papers-container:not(.hidden)", {
      timeout: 5000,
    });

    // Click on the cached paper (should be visible without searching)
    const cachedCard = page.locator(
      `#arxiv-papers-container > div:has(a[href*="${ATTENTION_PAPER.arxivId}"])`,
    );

    if (await cachedCard.isVisible()) {
      await cachedCard.click();

      // Should navigate directly to detail without showing modal
      await page.waitForURL("**/detail.html**", { timeout: 5000 });

      // Verify content loads
      await expect(page.locator("#paper-content:not(.hidden)")).toBeVisible();
    }
  });
});

/**
 * Tests for the Sleeper Agents paper - evaluate once, then run all related tests.
 * Uses sharedPage fixture to share browser context (and localStorage) across tests.
 */
test.describe.serial("Sleeper Agents Paper Tests", () => {
  test("evaluate and verify detail page", async ({ sharedPage: page }) => {
    // Clear cache and evaluate the paper
    await goToSearchPage(page, { clearCache: true });
    await evaluatePaper(page, SLEEPER_AGENTS_PAPER);

    const checker = new EvaluationChecker(SLEEPER_AGENTS_PAPER.title);

    // Extract page data
    const pageData = await extractDetailPageData(page);

    // Basic structure checks
    checker.checkNotEmpty("title exists", pageData.title);
    checker.checkEqual("year", pageData.year, SLEEPER_AGENTS_PAPER.expectedYear);
    checker.checkNotEmpty("novelty score exists", pageData.noveltyScore);
    checker.check("has paper summary", pageData.hasPaperSummary);
    checker.check("has conclusion", pageData.hasConclusion);
    checker.check("has graph", pageData.hasGraph);

    // Parse novelty score
    if (pageData.noveltyScore) {
      const scoreMatch = /^(\d)\/5$/.exec(pageData.noveltyScore);
      if (scoreMatch) {
        const score = parseInt(scoreMatch[1], 10);
        checker.checkInRange("novelty score in range", score, 1, 5);
      }
    }

    // Evidence limits
    checker.checkMaxLength(
      "supporting evidence max 5",
      pageData.supportingEvidenceCount,
      5,
    );
    checker.checkMaxLength(
      "contradictory evidence max 5",
      pageData.contradictoryEvidenceCount,
      5,
    );

    // LaTeX checks
    for (let i = 0; i < Math.min(3, pageData.evidenceTexts.length); i++) {
      checker.checkNoLatex(`evidence[${i}] no LaTeX`, pageData.evidenceTexts[i]);
      checker.checkNoBadSummaryPatterns(
        `evidence[${i}] no bad patterns`,
        pageData.evidenceTexts[i],
      );
    }

    // Extract cached data for deeper checks
    const cachedData = await extractCachedPaperData(page, SLEEPER_AGENTS_PAPER.arxivId);
    if (cachedData) {
      // Check date filtering
      for (const related of cachedData.relatedPapers) {
        checker.checkYearNotAfter(
          `related "${related.title.slice(0, 30)}..." year`,
          related.year,
          SLEEPER_AGENTS_PAPER.expectedYear,
        );
      }
    }

    checker.assertAllPassed();
  });

  test("detail page sections are collapsible", async ({ sharedPage: page }) => {
    // Navigate to cached paper detail page
    await goToCachedPaperDetail(page, SLEEPER_AGENTS_PAPER);

    // Check that sections exist and can be toggled
    const graphHeader = page.locator("#paper-graph-header");
    const graphContent = page.locator("#paper-graph");

    await expect(graphHeader).toBeVisible();
    await expect(graphContent).toBeVisible();

    // Click to collapse (graph starts expanded)
    await graphHeader.click();
    await page.waitForTimeout(300);
    await expect(graphContent).toBeHidden();

    // Click to expand again
    await graphHeader.click();
    await page.waitForTimeout(300);
    await expect(graphContent).toBeVisible();
  });

  test("cached paper navigates directly to detail", async ({ sharedPage: page }) => {
    // Navigate back to search
    await goToSearchPage(page);

    // Wait for cached papers to appear
    await page.waitForSelector("#arxiv-papers-container:not(.hidden)", {
      timeout: 5000,
    });

    // Click on the cached paper (should be visible without searching)
    const cachedCard = page.locator(
      `#arxiv-papers-container > div:has(a[href*="${SLEEPER_AGENTS_PAPER.arxivId}"])`,
    );

    if (await cachedCard.isVisible()) {
      await cachedCard.click();

      // Should navigate directly to detail without showing modal
      await page.waitForURL("**/detail.html**", { timeout: 5000 });

      // Verify content loads
      await expect(page.locator("#paper-content:not(.hidden)")).toBeVisible();
    }
  });
});
