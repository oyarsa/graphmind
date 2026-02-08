import { retryWithBackoff, waitForDOM, showInitError, cleanKeyword } from "../util";
import {
  AbstractEvaluationResponse,
  AbstractEvaluationResponseSchema,
  EvidenceItem,
  RelatedPaper,
} from "./model";
import {
  renderLatex,
  getRelationshipStyle,
  getRelatedPaperFilterType,
  findRelatedPaperIndexByTitle,
  setupSectionToggle,
  formatScientificCitation,
  getScoreDisplay,
  getScoreColor,
} from "./helpers";
import { addFooter } from "../footer";

/**
 * Load and display abstract evaluation details
 *
 * TODO: This duplicates some functionality from detail.ts - consider refactoring
 * shared components into a common module in a future iteration.
 */
function loadAbstractDetail(): void {
  const urlParams = new URLSearchParams(window.location.search);
  const evaluationId = urlParams.get("id");

  const loadingEl = document.getElementById("loading-detail");
  const contentEl = document.getElementById("paper-content");
  const errorEl = document.getElementById("error-detail");

  try {
    // Check if we have an evaluation ID
    if (!evaluationId) {
      throw new Error("No evaluation ID provided");
    }

    // Try to find the evaluation from cache
    console.log(`[AbstractDetail] Loading evaluation ${evaluationId} from cache`);
    const cachedEvaluation = localStorage.getItem(
      `abstract-evaluation-cache-${evaluationId}`,
    );

    if (!cachedEvaluation) {
      throw new Error("Evaluation not found in cache");
    }

    let evaluation: AbstractEvaluationResponse;
    try {
      const parsedData = JSON.parse(cachedEvaluation) as unknown;
      // Validate cached data against current schema
      evaluation = AbstractEvaluationResponseSchema.parse(parsedData);
      console.log(
        `[AbstractDetail] Cache validation successful for evaluation ${evaluationId}`,
      );
    } catch (schemaError) {
      // Schema validation failed - cache is outdated
      console.log(
        `[AbstractDetail] INVALID SCHEMA! Evaluation ${evaluationId} cache outdated, removing from cache...`,
      );
      console.warn(`[AbstractDetail] Schema validation error:`, schemaError);
      localStorage.removeItem(`abstract-evaluation-cache-${evaluationId}`);
      throw new Error("Cached evaluation data is invalid");
    }

    document.title = `${evaluation.title} - Abstract Evaluation`;

    // Populate the content
    displayAbstractEvaluation(evaluation);

    // Setup section toggles (same as regular detail page)
    setupSectionToggle("keywords");
    setupSectionToggle("related-papers");

    // Setup manual toggle for Novelty Assessment section
    setupNoveltyAssessmentToggle();

    // Show content and hide loading
    if (loadingEl) loadingEl.style.display = "none";
    if (contentEl) contentEl.classList.remove("hidden");
  } catch (error) {
    console.error("Error loading abstract evaluation details:", error);
    if (loadingEl) loadingEl.style.display = "none";
    if (errorEl) errorEl.classList.remove("hidden");
  }
}

/**
 * Display the abstract evaluation data in the UI
 */
function displayAbstractEvaluation(evaluation: AbstractEvaluationResponse): void {
  // Title and Abstract
  const titleEl = document.getElementById("paper-title");
  const abstractEl = document.getElementById("paper-abstract");

  if (titleEl) titleEl.innerHTML = renderLatex(evaluation.title);
  if (abstractEl) abstractEl.innerHTML = renderLatex(evaluation.abstract);

  // Keywords
  displayKeywords(evaluation.keywords);

  // Novelty Assessment section content
  displayNoveltyAssessment(evaluation);

  // Related Papers
  displayRelatedPapers(evaluation.related);
}

/**
 * Display extracted keywords as styled badges
 */
function displayKeywords(keywords: string[]): void {
  const keywordsContainer = document.getElementById("paper-keywords");
  if (!keywordsContainer) return;

  if (keywords.length === 0) {
    keywordsContainer.innerHTML = `
      <span class="text-gray-600 dark:text-gray-500 text-sm">
        No keywords extracted
      </span>
    `;
    return;
  }

  keywordsContainer.innerHTML = keywords
    .map(
      keyword => `
        <span class="px-3 py-1 bg-blue-100/70 dark:bg-blue-900/30 text-blue-800
                     dark:text-blue-300 text-sm rounded-md border border-blue-300/50
                     dark:border-blue-700/50 font-medium">
          ${cleanKeyword(keyword)}
        </span>
      `,
    )
    .join("");
}

/**
 * Display novelty assessment content (Paper Summary, Result, Supporting Evidence, Contradictory Evidence)
 * following the same structure as regular detail page
 */
function displayNoveltyAssessment(evaluation: AbstractEvaluationResponse): void {
  const evidenceContainer = document.getElementById("evidence-content");
  if (!evidenceContainer) return;

  // Create the structured evaluation display similar to regular detail page
  // Get rating (1-5) and determine color based on rating
  const rating = evaluation.label;
  const ratingDisplay = `${rating}/5`;

  // Color based on rating: 1-2 = red, 3 = yellow, 4-5 = green
  const noveltyBadgeColor =
    rating <= 2
      ? "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300 border-red-300 dark:border-red-700"
      : rating === 3
        ? "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300 border-yellow-300 dark:border-yellow-700"
        : "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300 border-green-300 dark:border-green-700";

  let evidenceHtml = `
    <div class="space-y-4">
      <!-- Result -->
      <div class="rounded-lg border border-gray-200 bg-gray-50/50 p-4
                  dark:border-gray-700 dark:bg-gray-800/50">
        <div class="mb-2 flex items-center gap-3">
          <div class="h-4 w-1 rounded-full bg-purple-500"></div>
          <h4 class="text-sm font-semibold tracking-wide text-gray-900 uppercase
                     dark:text-gray-100">
            Result
          </h4>
          <span class="px-2 py-0.5 text-xs font-medium rounded-md border ${noveltyBadgeColor}">
            Novelty: ${ratingDisplay}
          </span>
        </div>
        <p class="text-sm leading-relaxed text-gray-700 dark:text-gray-300">
          ${renderLatex(evaluation.conclusion)}
        </p>
      </div>

      <!-- Paper Summary -->
      <div class="rounded-lg border border-gray-200 bg-gray-50/50 p-4
                  dark:border-gray-700 dark:bg-gray-800/50">
        <div class="mb-2 flex items-center gap-2">
          <div class="h-4 w-1 rounded-full bg-blue-500"></div>
          <h4 class="text-sm font-semibold tracking-wide text-gray-900 uppercase
                     dark:text-gray-100">
            Paper Summary
          </h4>
        </div>
        <p class="text-sm leading-relaxed text-gray-700 dark:text-gray-300">
          ${renderLatex(evaluation.paper_summary)}
        </p>
      </div>
  `;

  // Key Comparisons (if available)
  if (evaluation.key_comparisons.length > 0) {
    evidenceHtml += `
      <div class="rounded-lg border border-gray-200 bg-gray-50/50 p-4
                  dark:border-gray-700 dark:bg-gray-800/50">
        <div class="mb-2 flex items-center gap-2">
          <div class="h-4 w-1 rounded-full bg-indigo-500"></div>
          <h4 class="text-sm font-semibold tracking-wide text-gray-900 uppercase
                     dark:text-gray-100">
            Key Comparisons
          </h4>
        </div>
        <ul class="space-y-2">
          ${evaluation.key_comparisons
            .map(
              comparison => `
            <li class="flex items-start gap-2">
              <span class="mt-1.5 block h-1.5 w-1.5 flex-shrink-0 rounded-full bg-indigo-500"></span>
              <span class="text-sm leading-relaxed text-gray-700 dark:text-gray-300">
                ${renderLatex(comparison)}
              </span>
            </li>
          `,
            )
            .join("")}
        </ul>
      </div>
    `;
  }

  // Supporting Evidence
  if (evaluation.supporting_evidence.length > 0) {
    evidenceHtml += `
      <div class="rounded-lg border border-gray-200 bg-gray-50/50 p-4
                  dark:border-gray-700 dark:bg-gray-800/50">
        <div class="mb-2 flex items-center gap-2">
          <div class="h-4 w-1 rounded-full bg-green-500"></div>
          <h4 class="text-sm font-semibold tracking-wide text-gray-900 uppercase
                     dark:text-gray-100">
            Supporting Evidence
          </h4>
        </div>
        <ul class="space-y-3">
          ${evaluation.supporting_evidence
            .map((evidence, index) =>
              createEvidenceItem(
                evidence,
                `supporting-${index}`,
                "bg-green-500",
                evaluation.related,
              ),
            )
            .join("")}
        </ul>
      </div>
    `;
  }

  // Contradictory Evidence
  if (evaluation.contradictory_evidence.length > 0) {
    evidenceHtml += `
      <div class="rounded-lg border border-gray-200 bg-gray-50/50 p-4
                  dark:border-gray-700 dark:bg-gray-800/50">
        <div class="mb-2 flex items-center gap-2">
          <div class="h-4 w-1 rounded-full bg-red-500"></div>
          <h4 class="text-sm font-semibold tracking-wide text-gray-900 uppercase
                     dark:text-gray-100">
            Contradictory Evidence
          </h4>
        </div>
        <ul class="space-y-3">
          ${evaluation.contradictory_evidence
            .map((evidence, index) =>
              createEvidenceItem(
                evidence,
                `contradictory-${index}`,
                "bg-red-500",
                evaluation.related,
              ),
            )
            .join("")}
        </ul>
      </div>
    `;
  }

  evidenceHtml += `
    </div>
  `;

  evidenceContainer.innerHTML = evidenceHtml;

  // Setup event handlers for related paper links and evidence expansion
  setupRelatedPaperLinkHandlers();
  setupEvidenceExpansionHandlers();
}

/**
 * Create an evidence item (either string or EvidenceItem object) - matches regular detail page format
 */
function createEvidenceItem(
  evidence: string | EvidenceItem,
  itemId: string,
  bulletColor: string,
  relatedPapers: RelatedPaper[],
): string {
  // Handle string evidence
  if (typeof evidence === "string") {
    return `
      <li class="flex items-start gap-2">
        <span class="mt-1.5 block h-1.5 w-1.5 flex-shrink-0 rounded-full ${bulletColor}"></span>
        <span class="text-sm leading-relaxed text-gray-700 dark:text-gray-300">
          ${renderLatex(evidence)}
        </span>
      </li>
    `;
  }

  // Handle EvidenceItem with paper information
  const relatedPaperIndex = evidence.paper_title
    ? findRelatedPaperIndexByTitle(evidence.paper_title, relatedPapers)
    : null;
  const relatedPaper =
    relatedPaperIndex !== null ? relatedPapers[relatedPaperIndex] : null;

  const displayText = relatedPaper
    ? formatScientificCitation(
        relatedPaper.authors,
        relatedPaper.year,
        evidence.paper_title ?? "Unknown Paper",
      )
    : (evidence.paper_title ?? "Unknown Paper");

  const paperTitleElement =
    relatedPaperIndex !== null
      ? `<a href="#related-papers"
           class="related-paper-link hover:underline cursor-pointer text-blue-800 dark:text-blue-200"
           data-paper-index="${relatedPaperIndex}">
           ${renderLatex(displayText)}:</a>`
      : `<span class="text-blue-800 dark:text-blue-200">
           ${renderLatex(displayText)}:</span>`;

  const hasExpandableContent =
    relatedPaper?.source === "semantic" &&
    ((relatedPaper.polarity === "positive" && Boolean(relatedPaper.target)) ||
      (relatedPaper.polarity === "negative" && Boolean(relatedPaper.background)));

  return `
    <li class="flex items-start gap-2">
      <span class="mt-1.5 block h-1.5 w-1.5 flex-shrink-0 rounded-full ${bulletColor}"></span>
      <div class="flex-1">
        <div class="text-sm leading-relaxed text-gray-700 dark:text-gray-300">
          <span class="font-medium">${paperTitleElement}</span> ${renderLatex(evidence.text)}
          ${
            hasExpandableContent
              ? `
            <button
              class="ml-2 text-xs text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-200 transition-colors duration-200 evidence-expand-btn"
              data-evidence-index="${itemId}"
              title="Show comparison details">
              <span class="expand-text">Show details</span>
              <svg class="inline-block w-3 h-3 ml-1 transform transition-transform duration-200 expand-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
              </svg>
            </button>
          `
              : ""
          }
        </div>
        ${
          hasExpandableContent
            ? `
          <div id="evidence-details-${itemId}"
               class="evidence-details hidden mt-3 pl-4 border-l-2 border-gray-200
                      dark:border-gray-700"
          >
            ${createEvidenceComparisonContent(relatedPaper)}
          </div>
        `
            : ""
        }
      </div>
    </li>
  `;
}

/**
 * Create comparison content for evidence details based on paper type
 */
function createEvidenceComparisonContent(relatedPaper: RelatedPaper): string {
  if (relatedPaper.source === "semantic") {
    // For semantic papers, show the related paper's background/target text
    if (relatedPaper.polarity === "negative" && relatedPaper.background) {
      return `
        <div class="rounded-lg border border-gray-200 bg-gray-50/50 p-3
                    dark:border-gray-700 dark:bg-gray-800/50">
          <div class="mb-2 flex items-center gap-2">
            <h5 class="text-xs font-semibold text-gray-900 uppercase dark:text-gray-100">
              Background
            </h5>
          </div>
          <p class="text-sm leading-relaxed text-gray-700 dark:text-gray-300">
            ${renderLatex(relatedPaper.background)}
          </p>
        </div>
      `;
    } else if (relatedPaper.polarity === "positive" && relatedPaper.target) {
      return `
        <div class="rounded-lg border border-gray-200 bg-gray-50/50 p-3
                    dark:border-gray-700 dark:bg-gray-800/50">
          <div class="mb-2 flex items-center gap-2">
            <h5 class="text-xs font-semibold text-gray-900 uppercase dark:text-gray-100">
              Target
            </h5>
          </div>
          <p class="text-sm leading-relaxed text-gray-700 dark:text-gray-300">
            ${renderLatex(relatedPaper.target)}
          </p>
        </div>
      `;
    }
  }

  // Fallback if no comparison data available
  return `
    <div class="text-center text-gray-600 dark:text-gray-400 py-2">
      No comparison data available.
    </div>
  `;
}

/**
 * Display related papers found during evaluation
 */
function displayRelatedPapers(relatedPapers: RelatedPaper[]): void {
  const countEl = document.getElementById("related-papers-count");
  const contentEl = document.getElementById("related-papers-content");

  if (countEl) {
    countEl.textContent = `(${relatedPapers.length} papers)`;
  }

  if (!contentEl) return;

  if (relatedPapers.length === 0) {
    contentEl.innerHTML = `
      <div class="text-center text-gray-600 dark:text-gray-400 py-8">
        No related papers found.
      </div>
    `;
    return;
  }

  // Setup filtering functionality
  setupRelatedPapersFiltering(relatedPapers);

  // Initial render with all papers visible
  renderFilteredRelatedPapers(
    relatedPapers,
    new Set(["background", "target", "supporting", "contrasting"]),
  );
}

/**
 * Create a card for a related paper
 */
function createRelatedPaperCard(paper: RelatedPaper, index: number): string {
  const { scorePercent } = getScoreDisplay(paper.score);
  const relationship = getRelationshipStyle(paper);

  return `
    <div
      id="related-card-${index}"
      class="rounded-lg border border-gray-300 bg-gray-50/50 p-4
             transition-all duration-200 hover:border-teal-500/50 dark:border-gray-700
             dark:bg-gray-800/50"
    >
      <div class="card-header cursor-pointer flex items-start justify-between">
        <div class="min-w-0 flex-1">
          <h4
            class="mb-2 line-clamp-2 font-semibold text-gray-900 dark:text-gray-100"
          >
            ${renderLatex(paper.title)}
          </h4>
          <div class="flex items-center gap-3">
            <!-- Relationship Type Badge -->
            <span
              class="${relationship.style} ${relationship.color} px-3 py-1 text-sm
                     font-medium whitespace-nowrap inline-flex items-center gap-1"
            >
              ${relationship.icon} ${relationship.label}
            </span>
            <div class="flex items-center gap-2">
              <span class="text-sm text-gray-600 dark:text-gray-400 whitespace-nowrap">
                Similarity Score: <span class="font-medium">${scorePercent}%</span>
              </span>
              <!-- Progress bar -->
              <div class="relative w-16 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                <div
                  class="absolute top-0 left-0 h-full rounded-full transition-all duration-300"
                  style="width: ${scorePercent}%; background-color: ${getScoreColor(paper.score)}"
                ></div>
              </div>
            </div>
          </div>
        </div>

        <div class="flex-shrink-0 ml-4 flex items-center gap-2">
          <!-- Go up button (shown when expanded) -->
          <button
            class="go-back-btn flex items-center gap-1 px-2 py-1 text-xs
                   bg-blue-100 text-blue-700 rounded-md hover:bg-blue-200
                   dark:bg-blue-900/30 dark:text-blue-300 dark:hover:bg-blue-900/50"
            title="Go up to Novelty Assessment section"
            style="display: none;"
          >
            <svg class="h-3 w-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19V5m0 0l-7 7m7-7l7 7"></path>
            </svg>
            Go up
          </button>

          <svg
            class="expand-icon h-5 w-5 text-gray-400 transition-transform duration-200"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M19 9l-7 7-7-7"
            ></path>
          </svg>
        </div>
      </div>

      <!-- Expanded Content -->
      <div
        id="expanded-content-${index}"
        class="mt-4 hidden border-t border-gray-200 pt-4 dark:border-gray-700"
      >
        <!-- Authors and Year -->
        ${
          paper.authors && paper.authors.length > 0 && paper.year
            ? `<div class="mb-4 text-sm text-gray-600 dark:text-gray-400">
               ${paper.authors.join(", ")} (${paper.year})
             </div>`
            : paper.authors && paper.authors.length > 0
              ? `<div class="mb-4 text-sm text-gray-600 dark:text-gray-400">
               ${paper.authors.join(", ")}
             </div>`
              : paper.year
                ? `<div class="mb-4 text-sm text-gray-600 dark:text-gray-400">
               (${paper.year})
             </div>`
                : ""
        }

        <div class="mb-4">
          <div class="mb-3 flex items-center gap-2">
            <div class="h-4 w-1 rounded-full bg-blue-500"></div>
            <h5
              class="text-sm font-semibold tracking-wide text-gray-900 uppercase
                     dark:text-gray-100"
            >
              Abstract
            </h5>
          </div>
          <p class="text-sm leading-relaxed text-gray-700 dark:text-gray-300">
            ${renderLatex(paper.abstract)}
          </p>
        </div>

        <div class="mb-4">
          <div class="mb-3 flex items-center gap-2">
            <div class="h-4 w-1 rounded-full bg-teal-500"></div>
            <h5
              class="text-sm font-semibold tracking-wide text-gray-900 uppercase
                     dark:text-gray-100"
            >
              Summary
            </h5>
          </div>
          <p class="text-sm leading-relaxed text-gray-700 dark:text-gray-300">
            ${renderLatex(paper.summary)}
          </p>
        </div>

        <!-- Text content for semantic papers -->
        ${
          paper.source === "semantic" && paper.polarity === "positive" && paper.target
            ? `<div class="mb-4">
                 <div class="mb-3 flex items-center gap-2">
                   <div class="h-4 w-1 rounded-full bg-orange-500"></div>
                   <h5 class="text-sm font-semibold tracking-wide text-gray-900 uppercase
                              dark:text-gray-100">
                     Target
                   </h5>
                 </div>
                 <p class="text-sm leading-relaxed text-gray-700 dark:text-gray-300">
                   ${renderLatex(paper.target)}
                 </p>
               </div>`
            : ""
        }
        ${
          paper.source === "semantic" &&
          paper.polarity === "negative" &&
          paper.background
            ? `<div class="mb-4">
                 <div class="mb-3 flex items-center gap-2">
                   <div class="h-4 w-1 rounded-full bg-green-500"></div>
                   <h5 class="text-sm font-semibold tracking-wide text-gray-900 uppercase
                              dark:text-gray-100">
                     Background
                   </h5>
                 </div>
                 <p class="text-sm leading-relaxed text-gray-700 dark:text-gray-300">
                   ${renderLatex(paper.background)}
                 </p>
               </div>`
            : ""
        }
      </div>
    </div>
  `;
}

/**
 * Setup filtering functionality for related papers
 */
function setupRelatedPapersFiltering(relatedPapers: RelatedPaper[]): void {
  const filtersContainer = document.getElementById("related-papers-filters");
  const filterChips = document.querySelectorAll(".filter-chip");
  const showAllButton = document.getElementById(
    "filter-show-all",
  ) as HTMLButtonElement | null;
  const hideAllButton = document.getElementById(
    "filter-hide-all",
  ) as HTMLButtonElement | null;

  if (!filtersContainer) return;

  // Calculate counts for each relationship type
  const countByType = (type: string) =>
    relatedPapers.filter(p => getRelatedPaperFilterType(p) === type).length;
  const counts = {
    background: countByType("background"),
    target: countByType("target"),
    supporting: countByType("supporting"),
    contrasting: countByType("contrasting"),
  };

  // Track active filters
  const activeFilters = new Set(["background", "target", "supporting", "contrasting"]);

  // Update filter chip counts and visibility
  const updateFilterCounts = () => {
    const chips = [
      { id: "#filter-background", count: counts.background },
      { id: "#filter-target", count: counts.target },
      { id: "#filter-supporting", count: counts.supporting },
      { id: "#filter-contrasting", count: counts.contrasting },
    ];

    chips.forEach(({ id, count }) => {
      const chip = document.querySelector(id);
      const countElement = document.querySelector(`${id} .filter-count`);

      if (countElement) {
        countElement.textContent = count.toString();
      }

      // Hide chips with zero count
      if (chip) {
        if (count === 0) {
          chip.classList.add("hidden");
          // Remove from active filters if it has no papers
          const type = chip.getAttribute("data-type");
          if (type) {
            activeFilters.delete(type);
          }
        } else {
          chip.classList.remove("hidden");
        }
      }
    });
  };

  // Update chip visual state
  const updateChipStates = () => {
    filterChips.forEach(chip => {
      const type = chip.getAttribute("data-type");
      if (type && activeFilters.has(type)) {
        // Active state - keep current colors but add border emphasis
        chip.classList.remove("opacity-50");
        chip.classList.add("ring-2", "ring-offset-1");
        if (type === "background") {
          chip.classList.add("ring-green-400", "dark:ring-green-600");
        } else if (type === "target") {
          chip.classList.add("ring-orange-400", "dark:ring-orange-600");
        } else if (type === "supporting") {
          chip.classList.add("ring-emerald-400", "dark:ring-emerald-600");
        } else if (type === "contrasting") {
          chip.classList.add("ring-red-400", "dark:ring-red-600");
        }
      } else {
        // Inactive state - dim and remove ring
        chip.classList.add("opacity-50");
        chip.classList.remove(
          "ring-2",
          "ring-offset-1",
          "ring-green-400",
          "ring-orange-400",
          "ring-emerald-400",
          "ring-red-400",
          "dark:ring-green-600",
          "dark:ring-orange-600",
          "dark:ring-emerald-600",
          "dark:ring-red-600",
        );
      }
    });

    // Update Show All/Hide All button states
    // Only consider filter types that have papers available
    const availableFilterTypes = [
      counts.background > 0 ? "background" : null,
      counts.target > 0 ? "target" : null,
      counts.supporting > 0 ? "supporting" : null,
      counts.contrasting > 0 ? "contrasting" : null,
    ].filter(type => type !== null);

    const allFiltersActive =
      availableFilterTypes.length > 0 &&
      availableFilterTypes.every(type => activeFilters.has(type));
    const noFiltersActive = activeFilters.size === 0;

    if (showAllButton) {
      showAllButton.disabled = allFiltersActive;
      if (allFiltersActive) {
        showAllButton.classList.add("opacity-50", "cursor-not-allowed");
      } else {
        showAllButton.classList.remove("opacity-50", "cursor-not-allowed");
      }
    }

    if (hideAllButton) {
      hideAllButton.disabled = noFiltersActive;
      if (noFiltersActive) {
        hideAllButton.classList.add("opacity-50", "cursor-not-allowed");
      } else {
        hideAllButton.classList.remove("opacity-50", "cursor-not-allowed");
      }
    }
  };

  // Add click handlers for filter chips
  filterChips.forEach(chip => {
    chip.addEventListener("click", () => {
      const type = chip.getAttribute("data-type");
      if (!type) return;

      if (activeFilters.has(type)) {
        activeFilters.delete(type);
      } else {
        activeFilters.add(type);
      }

      updateChipStates();
      renderFilteredRelatedPapers(relatedPapers, activeFilters);
    });
  });

  // Add click handler for show all button
  if (showAllButton) {
    showAllButton.addEventListener("click", () => {
      activeFilters.clear();
      // Only add filters that have papers available
      if (counts.background > 0) activeFilters.add("background");
      if (counts.target > 0) activeFilters.add("target");
      if (counts.supporting > 0) activeFilters.add("supporting");
      if (counts.contrasting > 0) activeFilters.add("contrasting");

      updateChipStates();
      renderFilteredRelatedPapers(relatedPapers, activeFilters);
    });
  }

  // Add click handler for hide all button
  if (hideAllButton) {
    hideAllButton.addEventListener("click", () => {
      activeFilters.clear();

      updateChipStates();
      renderFilteredRelatedPapers(relatedPapers, activeFilters);
    });
  }

  // Initial setup - update counts first, then chip states
  updateFilterCounts();
  updateChipStates();
}

/**
 * Render related papers based on active filters
 */
function renderFilteredRelatedPapers(
  relatedPapers: RelatedPaper[],
  activeFilters: Set<string>,
): void {
  const relatedPapersContainer = document.getElementById("related-papers-content");
  if (!relatedPapersContainer) return;

  // Filter papers based on their relationship type
  const filteredPapers = relatedPapers.filter(paper => {
    const filterType = getRelatedPaperFilterType(paper);
    return filterType !== null && activeFilters.has(filterType);
  });

  if (filteredPapers.length === 0) {
    relatedPapersContainer.innerHTML = `
      <div class="text-gray-600 dark:text-gray-500 text-sm text-center py-4">
        No related papers match the selected filters.
      </div>
    `;
    return;
  }

  // Render filtered papers
  relatedPapersContainer.innerHTML = filteredPapers
    .map((paper, index) => createRelatedPaperCard(paper, index))
    .join("");

  // Add click handlers for expanding papers
  filteredPapers.forEach((_, index) => {
    const cardHeader = document.querySelector(`#related-card-${index} .card-header`);
    const expandedContent = document.getElementById(`expanded-content-${index}`);
    const goBackBtn = cardHeader?.querySelector(".go-back-btn") as HTMLElement | null;

    if (cardHeader && expandedContent) {
      // Handle card expansion/collapse
      cardHeader.addEventListener("click", event => {
        // Prevent expansion if clicking the go back button
        if ((event.target as Element).closest(".go-back-btn")) {
          return;
        }

        const isExpanded = !expandedContent.classList.contains("hidden");
        expandedContent.classList.toggle("hidden", isExpanded);
        cardHeader
          .querySelector(".expand-icon")
          ?.classList.toggle("rotate-180", !isExpanded);

        // Show/hide go back button based on expansion state
        updateGoBackButtonVisibility(goBackBtn, !isExpanded);
      });

      // Handle go back button click
      if (goBackBtn) {
        goBackBtn.addEventListener("click", event => {
          event.stopPropagation(); // Prevent card expansion
          scrollToNoveltyAssessmentSection();
        });
      }

      // Set initial go back button visibility
      const isInitiallyExpanded = !expandedContent.classList.contains("hidden");
      updateGoBackButtonVisibility(goBackBtn, isInitiallyExpanded);
    }
  });
}

/**
 * Update the visibility of the go back button based on expansion state
 */
function updateGoBackButtonVisibility(
  goBackBtn: HTMLElement | null,
  isExpanded: boolean,
): void {
  if (!goBackBtn) return;

  if (isExpanded) {
    // Show button always for expanded cards
    goBackBtn.style.display = "flex";
  } else {
    // Hide button for collapsed cards
    goBackBtn.style.display = "none";
  }
}

/**
 * Scroll to the Novelty Assessment section
 */
function scrollToNoveltyAssessmentSection(): void {
  const assessmentHeader = document.getElementById("structured-evaluation-header");

  if (assessmentHeader) {
    assessmentHeader.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

/**
 * Setup event handlers for related paper links
 */
function setupRelatedPaperLinkHandlers(): void {
  document.addEventListener("click", event => {
    const target = event.target as HTMLElement;
    if (target.classList.contains("related-paper-link")) {
      event.preventDefault();

      // First, enable all paper types to ensure the target paper is visible
      const showAllButton = document.getElementById(
        "filter-show-all",
      ) as HTMLButtonElement | null;
      if (showAllButton) {
        showAllButton.click();
      }

      const paperIndex = parseInt(target.getAttribute("data-paper-index") ?? "-1");
      if (paperIndex >= 0) {
        // Scroll to and expand the corresponding related paper card
        const relatedCard = document.getElementById(`related-card-${paperIndex}`);
        const expandedContent = document.getElementById(
          `expanded-content-${paperIndex}`,
        );
        const goBackBtn = relatedCard?.querySelector(
          ".go-back-btn",
        ) as HTMLElement | null;

        if (relatedCard && expandedContent) {
          // Show the expanded content
          expandedContent.classList.remove("hidden");
          const expandIcon = relatedCard.querySelector(".expand-icon");
          if (expandIcon) {
            expandIcon.classList.add("rotate-180");
          }

          // Show the go back button when expanded
          updateGoBackButtonVisibility(goBackBtn, true);

          // Scroll to the card
          relatedCard.scrollIntoView({ behavior: "smooth", block: "center" });
        }
      } else {
        // Just scroll to Related Papers section if no specific paper
        const relatedPapersHeader = document.getElementById("related-papers-header");
        if (relatedPapersHeader) {
          relatedPapersHeader.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      }
    }
  });
}

/**
 * Setup event handlers for evidence expansion buttons
 */
function setupEvidenceExpansionHandlers(): void {
  document.addEventListener("click", event => {
    const target = event.target as HTMLElement;

    // Find the button element (could be the target itself or a parent)
    const button = target.closest(".evidence-expand-btn");

    if (button) {
      const evidenceIndex = button.getAttribute("data-evidence-index");

      if (evidenceIndex) {
        const detailsEl = document.getElementById(`evidence-details-${evidenceIndex}`);

        if (detailsEl) {
          const isHidden = detailsEl.classList.contains("hidden");
          const expandText = button.querySelector(".expand-text");
          const expandIcon = button.querySelector(".expand-icon");

          if (isHidden) {
            detailsEl.classList.remove("hidden");
            if (expandText) expandText.textContent = "Hide details";
            if (expandIcon) expandIcon.classList.add("rotate-180");
          } else {
            detailsEl.classList.add("hidden");
            if (expandText) expandText.textContent = "Show details";
            if (expandIcon) expandIcon.classList.remove("rotate-180");
          }
        }
      }
    }
  });
}

/**
 * Setup toggle functionality for Novelty Assessment section
 */
function setupNoveltyAssessmentToggle(): void {
  const header = document.getElementById("structured-evaluation-header");
  const content = document.getElementById("evidence-content");
  const chevron = document.getElementById("structured-evaluation-chevron");

  if (header && content && chevron) {
    header.addEventListener("click", () => {
      const isHidden = content.classList.contains("hidden");

      if (isHidden) {
        content.classList.remove("hidden");
        chevron.style.transform = "rotate(180deg)";
      } else {
        content.classList.add("hidden");
        chevron.style.transform = "rotate(0deg)";
      }
    });

    // Start with content visible
    content.classList.remove("hidden");
    chevron.style.transform = "rotate(180deg)";
  }
}

/**
 * Reset all collapsible sections to their default expanded state.
 * This is called when the page is restored from bfcache to ensure
 * sections don't retain their collapsed state from previous visits.
 */
function resetSectionStates(): void {
  // Standard sections using setupSectionToggle
  const standardSections = ["keywords", "related-papers"];
  for (const baseId of standardSections) {
    const content = document.getElementById(baseId);
    const chevron = document.getElementById(`${baseId}-chevron`);

    if (content && chevron) {
      content.classList.remove("hidden");
      chevron.classList.add("rotate-180");
    }
  }

  // Novelty Assessment section (uses different structure)
  const evidenceContent = document.getElementById("evidence-content");
  const evalChevron = document.getElementById("structured-evaluation-chevron");
  if (evidenceContent && evalChevron) {
    evidenceContent.classList.remove("hidden");
    evalChevron.style.transform = "rotate(180deg)";
  }
}

/**
 * Initialize the Abstract Detail application.
 */
async function initializeApp(): Promise<void> {
  await waitForDOM();

  // Handle bfcache restoration - reset sections to expanded state
  window.addEventListener("pageshow", event => {
    if (event.persisted) {
      resetSectionStates();
    }
  });

  await retryWithBackoff(() => {
    loadAbstractDetail();
    return Promise.resolve();
  });

  // Add footer
  addFooter();
}

initializeApp().catch(showInitError);
