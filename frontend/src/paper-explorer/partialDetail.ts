import {
  retryWithBackoff,
  waitForDOM,
  showInitError,
  isMobileDevice,
  showMobileMessage,
} from "../util";
import {
  PartialEvaluationResponse,
  PartialEvaluationResponseSchema,
  EvidenceItem,
  RelatedPaper,
} from "./model";
import {
  renderLatex,
  getRelationshipStyle,
  setupSectionToggle,
  createSideBySideComparison,
} from "./helpers";
import { addFooter } from "../footer";

/**
 * Load and display partial evaluation details
 *
 * TODO: This duplicates some functionality from detail.ts - consider refactoring
 * shared components into a common module in a future iteration.
 */
function loadPartialDetail(): void {
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
    console.log(`[PartialDetail] Loading evaluation ${evaluationId} from cache`);
    const cachedEvaluation = localStorage.getItem(
      `partial-evaluation-cache-${evaluationId}`,
    );

    if (!cachedEvaluation) {
      throw new Error("Evaluation not found in cache");
    }

    let evaluation: PartialEvaluationResponse;
    try {
      const parsedData = JSON.parse(cachedEvaluation) as unknown;
      // Validate cached data against current schema
      evaluation = PartialEvaluationResponseSchema.parse(parsedData);
      console.log(
        `[PartialDetail] Cache validation successful for evaluation ${evaluationId}`,
      );
    } catch (schemaError) {
      // Schema validation failed - cache is outdated
      console.log(
        `[PartialDetail] INVALID SCHEMA! Evaluation ${evaluationId} cache outdated, removing from cache...`,
      );
      console.warn(`[PartialDetail] Schema validation error:`, schemaError);
      localStorage.removeItem(`partial-evaluation-cache-${evaluationId}`);
      throw new Error("Cached evaluation data is invalid");
    }

    document.title = `${evaluation.title} - Abstract Evaluation`;

    // Populate the content
    displayPartialEvaluation(evaluation);

    // Setup section toggles (same as regular detail page)
    setupSectionToggle("keywords");
    setupSectionToggle("related-papers");

    // Setup manual toggle for Novelty Assessment section
    setupNoveltyAssessmentToggle();

    // Show content and hide loading
    if (loadingEl) loadingEl.style.display = "none";
    if (contentEl) contentEl.classList.remove("hidden");
  } catch (error) {
    console.error("Error loading partial evaluation details:", error);
    if (loadingEl) loadingEl.style.display = "none";
    if (errorEl) errorEl.classList.remove("hidden");
  }
}

/**
 * Display the partial evaluation data in the UI
 */
function displayPartialEvaluation(evaluation: PartialEvaluationResponse): void {
  // Title and Abstract
  const titleEl = document.getElementById("paper-title");
  const abstractEl = document.getElementById("paper-abstract");

  if (titleEl) titleEl.innerHTML = renderLatex(evaluation.title);
  if (abstractEl) abstractEl.innerHTML = renderLatex(evaluation.abstract);

  // Keywords
  displayKeywords(evaluation.keywords);

  // Display novelty rating in the top section (if exists)
  displayNoveltyRating(evaluation);

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
          ${keyword}
        </span>
      `,
    )
    .join("");
}

/**
 * Display novelty rating in the top section (similar to regular Detail page)
 */
function displayNoveltyRating(evaluation: PartialEvaluationResponse): void {
  const ratingEl = document.getElementById("paper-rating");

  if (ratingEl && evaluation.probability != null) {
    const probability = evaluation.probability;
    const probabilityPercent = Math.round(probability * 100);
    const novelText = `${probabilityPercent}%`;
    const novelClass =
      probability >= 0.5
        ? "text-green-600 dark:text-green-400 font-semibold"
        : "text-red-600 dark:text-red-400 font-semibold";

    ratingEl.textContent = novelText;
    ratingEl.className = novelClass;
  } else if (ratingEl) {
    // Fallback to binary label display
    const isNovel = evaluation.label === 1;
    const novelText = isNovel ? "High" : "Low";
    const novelClass = isNovel
      ? "text-green-600 dark:text-green-400 font-semibold"
      : "text-red-600 dark:text-red-400 font-semibold";

    ratingEl.textContent = novelText;
    ratingEl.className = novelClass;
  }
}

/**
 * Display novelty assessment content (Paper Summary, Result, Supporting Evidence, Contradictory Evidence)
 * following the same structure as regular detail page
 */
function displayNoveltyAssessment(evaluation: PartialEvaluationResponse): void {
  const evidenceContainer = document.getElementById("evidence-content");
  if (!evidenceContainer) return;

  // Create the structured evaluation display similar to regular detail page
  let evidenceHtml = `
    <div class="space-y-4">
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

      <!-- Result -->
      <div class="rounded-lg border border-gray-200 bg-gray-50/50 p-4
                  dark:border-gray-700 dark:bg-gray-800/50">
        <div class="mb-2 flex items-center gap-2">
          <div class="h-4 w-1 rounded-full bg-purple-500"></div>
          <h4 class="text-sm font-semibold tracking-wide text-gray-900 uppercase
                     dark:text-gray-100">
            Result
          </h4>
        </div>
        <p class="text-sm leading-relaxed text-gray-700 dark:text-gray-300">
          ${renderLatex(evaluation.conclusion)}
        </p>
      </div>
  `;

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

  // Setup event handlers for related paper links
  setupRelatedPaperLinkHandlers();
}

/**
 * Create an evidence item (either string or EvidenceItem object)
 */
function createEvidenceItem(
  evidence: string | EvidenceItem,
  _itemId: string,
  accentColor: string,
  relatedPapers: RelatedPaper[],
): string {
  if (typeof evidence === "string") {
    return `
      <li class="flex items-start gap-2">
        <div class="mt-1.5 h-2 w-2 rounded-full ${accentColor}"></div>
        <span class="text-sm leading-relaxed text-gray-700 dark:text-gray-300 flex-1">
          ${renderLatex(evidence)}
        </span>
      </li>
    `;
  }

  // Evidence is an EvidenceItem object
  // Find the related paper for this evidence item
  const relatedPaper = relatedPapers.find(
    paper => paper.title === evidence.paper_title,
  );
  const relatedPaperIndex = relatedPaper ? relatedPapers.indexOf(relatedPaper) : -1;

  return `
    <li class="flex items-start gap-2">
      <div class="mt-1.5 h-2 w-2 rounded-full ${accentColor}"></div>
      <div class="flex-1">
        <span class="text-sm leading-relaxed text-gray-700 dark:text-gray-300">
          ${renderLatex(evidence.text)}
        </span>
        ${
          evidence.paper_title && relatedPaperIndex >= 0
            ? `
          <div class="mt-1 text-xs">
            Source: <button 
              class="related-paper-link text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 underline cursor-pointer"
              data-paper-index="${relatedPaperIndex}"
            >
              ${evidence.paper_title}
            </button>
          </div>
        `
            : evidence.paper_title
              ? `
          <div class="mt-1 text-xs text-gray-500 dark:text-gray-500">
            Source: ${evidence.paper_title}
          </div>
        `
              : ""
        }
      </div>
    </li>
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

  contentEl.innerHTML = relatedPapers
    .map((paper, index) => createRelatedPaperCard(paper, index))
    .join("");

  // Add click handlers for expanding papers
  relatedPapers.forEach((_, index) => {
    const cardHeader = document.querySelector(`#related-card-${index} .card-header`);
    const expandedContent = document.getElementById(`expanded-content-${index}`);

    if (cardHeader && expandedContent) {
      cardHeader.addEventListener("click", () => {
        const isExpanded = !expandedContent.classList.contains("hidden");
        expandedContent.classList.toggle("hidden", isExpanded);
        cardHeader
          .querySelector(".expand-icon")
          ?.classList.toggle("rotate-180", !isExpanded);
      });
    }
  });
}

/**
 * Create a card for a related paper
 */
function createRelatedPaperCard(paper: RelatedPaper, index: number): string {
  const relationship = getRelationshipStyle(paper);
  const scorePercent = Math.round(paper.score * 100);

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
                     font-medium whitespace-nowrap"
            >
              ${relationship.icon} ${relationship.label}
            </span>
            <div class="flex items-center gap-2">
              <span class="text-sm text-gray-600 dark:text-gray-400 whitespace-nowrap">
                Similarity: <span class="font-medium">${scorePercent}%</span>
              </span>
            </div>
          </div>
        </div>

        <div class="flex-shrink-0 ml-4 flex items-center gap-2">
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

        <!-- Text Comparisons for semantic papers -->
        ${
          paper.source === "semantic" && paper.polarity === "positive" && paper.target
            ? `<div class="mb-4">
                 ${createSideBySideComparison(
                   "Main Paper",
                   null, // Partial evaluation doesn't have main paper target text
                   "Related Paper",
                   paper.target,
                   "target",
                 )}
               </div>`
            : ""
        }
        ${
          paper.source === "semantic" &&
          paper.polarity === "negative" &&
          paper.background
            ? `<div class="mb-4">
                 ${createSideBySideComparison(
                   "Main Paper",
                   null, // Partial evaluation doesn't have main paper background text
                   "Related Paper",
                   paper.background,
                   "background",
                 )}
               </div>`
            : ""
        }
      </div>
    </div>
  `;
}

/**
 * Setup event handlers for related paper links
 */
function setupRelatedPaperLinkHandlers(): void {
  document.addEventListener("click", event => {
    const target = event.target as HTMLElement;
    if (target.classList.contains("related-paper-link")) {
      const paperIndex = parseInt(target.getAttribute("data-paper-index") ?? "-1");
      if (paperIndex >= 0) {
        // Scroll to and expand the corresponding related paper card
        const relatedCard = document.getElementById(`related-card-${paperIndex}`);
        const expandedContent = document.getElementById(
          `expanded-content-${paperIndex}`,
        );

        if (relatedCard && expandedContent) {
          // Show the expanded content
          expandedContent.classList.remove("hidden");
          const expandIcon = relatedCard.querySelector(".expand-icon");
          if (expandIcon) {
            expandIcon.classList.add("rotate-180");
          }

          // Scroll to the card
          relatedCard.scrollIntoView({ behavior: "smooth", block: "center" });
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
 * Initialize the Partial Detail application.
 */
async function initializeApp(): Promise<void> {
  await waitForDOM();

  if (isMobileDevice()) {
    showMobileMessage();
    return;
  }

  await retryWithBackoff(() => {
    loadPartialDetail();
    return Promise.resolve();
  });

  // Add footer
  addFooter();
}

initializeApp().catch(showInitError);
