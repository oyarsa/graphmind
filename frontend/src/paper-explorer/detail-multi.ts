import {
  retryWithBackoff,
  waitForDOM,
  showInitError,
  isMobileDevice,
  showMobileMessage,
  cleanKeyword,
} from "../util";
import {
  GraphResultMulti,
  GraphResultMultiSchema,
  RelatedPaper,
  Entity,
  PerspectiveEvaluation,
  MultiEvaluationResult,
  StructuredEval,
} from "./model";
import {
  createPaperTermsDisplay,
  getRelationshipStyle,
  getScoreDisplay,
  setupSectionToggle,
  renderLatex,
  getArxivUrl,
  formatConferenceName,
  createSideBySideComparison,
  createHierarchicalGraph,
  createExpandableEvidenceItem,
  findRelatedPaperIndexByTitle,
  getRelatedPaperFilterType,
} from "./helpers";
import { addFooter } from "../footer";
import { buildPageUrl } from "./routes";

function createRelatedPaperCard(
  paper: RelatedPaper,
  index: number,
  mainPaperBackground?: string | null,
): string {
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
        <h4 class="mb-2 line-clamp-2 pr-2 text-base font-semibold text-gray-900
           dark:text-gray-100">
          ${renderLatex(paper.title)}
        </h4>
        <p class="mb-3 text-sm text-gray-600 dark:text-gray-400">
          ${paper.authors?.join(", ") ?? "Unknown authors"} • ${paper.year ?? "Unknown year"}
        </p>
        <div class="mb-3 flex flex-wrap items-center gap-2">
          <div class="${relationship.color} rounded-md px-2 py-1 inline-flex items-center gap-1">
            <span class="text-xs font-medium inline-flex items-center gap-1">${relationship.icon} ${relationship.label}</span>
          </div>
          <div class="text-xs text-gray-600 dark:text-gray-400">
            ${scorePercent}% similarity
          </div>
        </div>
      </div>
      <div class="ml-2 flex items-center">
        <svg class="h-5 w-5 text-gray-400 dark:text-gray-600" fill="none"
             stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                d="M19 9l-7 7-7-7"></path>
        </svg>
      </div>
    </div>
    <div class="card-content hidden mt-4 space-y-3">
      <div>
        <h5 class="mb-2 text-sm font-medium text-gray-800 dark:text-gray-200">
          Summary
        </h5>
        <p class="text-sm leading-relaxed text-gray-700 dark:text-gray-300">
          ${renderLatex(paper.summary)}
        </p>
      </div>
      ${
        (paper.background ?? paper.target)
          ? createSideBySideComparison(
              "Main Paper",
              mainPaperBackground ?? null,
              "Related Paper",
              paper.background ?? paper.target ?? null,
              paper.background ? "background" : "target",
            )
          : ""
      }
    </div>
  </div>`;
}

function createPerspectiveCard(perspective: PerspectiveEvaluation): string {
  const noveltyText = perspective.label === 1 ? "Novel" : "Not Novel";
  const noveltyColor =
    perspective.label === 1
      ? "text-green-600 dark:text-green-400"
      : "text-red-600 dark:text-red-400";

  return `
  <div class="rounded-lg border border-gray-300 bg-gray-50/50 p-4
             dark:border-gray-700 dark:bg-gray-800/50">
    <div class="flex items-start justify-between mb-3">
      <h4 class="text-base font-semibold text-gray-900 dark:text-gray-100">
        📊 ${perspective.name.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase())}
      </h4>
      <div class="flex items-center">
        <span class="text-sm font-medium ${noveltyColor}">
          ${noveltyText}
        </span>
      </div>
    </div>
    <div class="text-sm text-gray-700 dark:text-gray-300 leading-relaxed">
      ${renderLatex(perspective.rationale)}
    </div>
  </div>`;
}

function displayMultiPerspectiveEvaluation(evaluation: MultiEvaluationResult): void {
  const individualPerspectives = document.getElementById("individual-perspectives");
  const overallRationale = document.getElementById("overall-rationale");
  const overallLabel = document.getElementById("overall-label");
  const overallProbability = document.getElementById("overall-probability");

  if (
    !individualPerspectives ||
    !overallRationale ||
    !overallLabel ||
    !overallProbability
  ) {
    console.error("Multi-perspective evaluation elements not found");
    return;
  }

  // Display individual perspectives
  individualPerspectives.innerHTML = evaluation.perspectives
    .map(perspective => createPerspectiveCard(perspective))
    .join("");

  // Display overall assessment
  overallRationale.textContent = evaluation.rationale;

  const noveltyText = evaluation.label === 1 ? "Novel" : "Not Novel";
  const noveltyColor =
    evaluation.label === 1
      ? "text-green-600 dark:text-green-400"
      : "text-red-600 dark:text-red-400";

  overallLabel.className = `text-lg font-bold ${noveltyColor}`;
  overallLabel.textContent = noveltyText;

  const probabilityPercent = Math.round(evaluation.probability * 100);
  overallProbability.textContent = `${probabilityPercent}%`;
}

function createStructuredAnalysisDisplay(
  structured: StructuredEval,
  graphResult: GraphResultMulti | null,
): string {
  return `
    <div class="space-y-4">
      <!-- Summary -->
      <div class="rounded-lg border border-gray-200 bg-gray-50/50 p-4
                  dark:border-gray-700 dark:bg-gray-800/50">
        <div class="mb-2 flex items-center gap-3">
          <div class="h-4 w-1 rounded-full bg-blue-500"></div>
          <h4 class="text-sm font-semibold tracking-wide text-gray-900 uppercase
                     dark:text-gray-100">
            Paper Summary
          </h4>
        </div>
        <p class="text-sm leading-relaxed text-gray-700 dark:text-gray-300">
          ${renderLatex(structured.paper_summary)}
        </p>
      </div>

      <!-- Result -->
      <div class="rounded-lg border border-gray-200 bg-gray-50/50 p-4
                  dark:border-gray-700 dark:bg-gray-800/50">
        <div class="mb-2 flex items-center gap-3">
          <div class="h-4 w-1 rounded-full bg-purple-500"></div>
          <h4 class="text-sm font-semibold tracking-wide text-gray-900 uppercase
                     dark:text-gray-100">
            Conclusion
          </h4>
        </div>
        <p class="text-sm leading-relaxed text-gray-700 dark:text-gray-300">
          ${renderLatex(structured.conclusion)}
        </p>
      </div>

      <!-- Supporting Evidence -->
      ${
        structured.supporting_evidence.length > 0
          ? `
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
            ${structured.supporting_evidence
              .map((evidence, index) => {
                const relatedPaperIndex =
                  typeof evidence === "object" && evidence.paper_title && graphResult
                    ? findRelatedPaperIndexByTitle(
                        evidence.paper_title,
                        graphResult.related,
                      )
                    : null;
                const relatedPaper =
                  relatedPaperIndex !== null && graphResult
                    ? graphResult.related[relatedPaperIndex]
                    : null;

                return createExpandableEvidenceItem(
                  evidence,
                  `supporting-multi-${index}`,
                  relatedPaper,
                  relatedPaperIndex,
                  "bg-green-500",
                );
              })
              .join("")}
          </ul>
        </div>
      `
          : ""
      }

      <!-- Contradictory Evidence -->
      ${
        structured.contradictory_evidence.length > 0
          ? `
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
            ${structured.contradictory_evidence
              .map((evidence, index) => {
                const relatedPaperIndex =
                  typeof evidence === "object" && evidence.paper_title && graphResult
                    ? findRelatedPaperIndexByTitle(
                        evidence.paper_title,
                        graphResult.related,
                      )
                    : null;
                const relatedPaper =
                  relatedPaperIndex !== null && graphResult
                    ? graphResult.related[relatedPaperIndex]
                    : null;

                return createExpandableEvidenceItem(
                  evidence,
                  `contradictory-multi-${index}`,
                  relatedPaper,
                  relatedPaperIndex,
                  "bg-red-500",
                );
              })
              .join("")}
          </ul>
        </div>
      `
          : ""
      }
    </div>
  `;
}

function showExcerptsModal(entity: Entity): void {
  const modal = document.getElementById("excerptsModal");
  const modalTitle = document.getElementById("modalTitle");
  const modalContent = document.getElementById("modalContent");
  const closeButton = document.getElementById("closeModal");

  if (!modal || !modalTitle || !modalContent || !closeButton) return;

  // Set modal title
  modalTitle.innerHTML = renderLatex(entity.label);

  // Format excerpts
  if (entity.excerpts && entity.excerpts.length > 0) {
    modalContent.innerHTML = entity.excerpts
      .map(
        excerpt => `
        <div class="rounded-lg border border-gray-200 bg-gray-50 p-4
                    dark:border-gray-700 dark:bg-gray-800">
          <div class="mb-2 text-xs font-semibold text-gray-500 dark:text-gray-400">
            ${excerpt.section}
          </div>
          <div class="text-sm text-gray-700 dark:text-gray-300">
            ${renderLatex(excerpt.text)}
          </div>
        </div>`,
      )
      .join("");
  } else {
    modalContent.innerHTML = `
      <div class="text-center text-gray-500 dark:text-gray-400">
        No excerpts available for this entity.
      </div>`;
  }

  // Show modal
  modal.classList.remove("hidden");
  modal.classList.add("flex");

  // Close button handler
  const closeModal = () => {
    modal.classList.add("hidden");
    modal.classList.remove("flex");
  };

  closeButton.onclick = closeModal;

  // Click outside to close
  modal.onclick = e => {
    if (e.target === modal) {
      closeModal();
    }
  };

  // ESC key to close
  const handleEsc = (e: KeyboardEvent) => {
    if (e.key === "Escape") {
      closeModal();
      document.removeEventListener("keydown", handleEsc);
    }
  };
  document.addEventListener("keydown", handleEsc);
}

function displayRelatedPapers(
  relatedPapers: RelatedPaper[],
  backgroundText?: string | null,
): void {
  const container = document.getElementById("related-papers-content");
  if (!container) {
    console.error("Related papers container not found");
    return;
  }

  // Group papers by relationship type
  const backgroundPapers = relatedPapers.filter(
    p => getRelatedPaperFilterType(p) === "background",
  );
  const targetPapers = relatedPapers.filter(
    p => getRelatedPaperFilterType(p) === "target",
  );
  const supportingPapers = relatedPapers.filter(
    p => getRelatedPaperFilterType(p) === "supporting",
  );
  const contrastingPapers = relatedPapers.filter(
    p => getRelatedPaperFilterType(p) === "contrasting",
  );

  // Update filter counts
  const updateFilterCount = (id: string, count: number) => {
    const filterElement = document.getElementById(id);
    if (filterElement) {
      const countSpan = filterElement.querySelector(".filter-count");
      if (countSpan) {
        countSpan.textContent = count.toString();
      }
    }
  };

  updateFilterCount("filter-background", backgroundPapers.length);
  updateFilterCount("filter-target", targetPapers.length);
  updateFilterCount("filter-supporting", supportingPapers.length);
  updateFilterCount("filter-contrasting", contrastingPapers.length);

  // Create paper cards
  const paperCards = relatedPapers
    .map((paper, index) => createRelatedPaperCard(paper, index, backgroundText))
    .join("");

  container.innerHTML = paperCards;

  // Add click handlers for expandable cards
  relatedPapers.forEach((_, index) => {
    const card = document.getElementById(`related-card-${index}`);
    if (card) {
      const header = card.querySelector(".card-header");
      const content = card.querySelector(".card-content");
      const chevron = card.querySelector("svg");

      if (header && content && chevron) {
        header.addEventListener("click", () => {
          const isHidden = content.classList.contains("hidden");
          content.classList.toggle("hidden");
          chevron.style.transform = isHidden ? "rotate(180deg)" : "rotate(0deg)";
        });
      }
    }
  });

  // Set up filter functionality (reuse existing logic)
  setupRelatedPapersFilters(relatedPapers);
}

function setupRelatedPapersFilters(relatedPapers: RelatedPaper[]): void {
  const filterButtons = document.querySelectorAll(".filter-chip");
  const showAllButton = document.getElementById("filter-show-all");
  const hideAllButton = document.getElementById("filter-hide-all");

  filterButtons.forEach(button => {
    button.addEventListener("click", () => {
      const type = button.getAttribute("data-type");
      if (type) {
        filterPapersByType(type, relatedPapers);
        updateFilterButtonStates(button as HTMLElement);
      }
    });
  });

  showAllButton?.addEventListener("click", () => {
    showAllPapers();
    clearFilterButtonStates();
  });

  hideAllButton?.addEventListener("click", () => {
    hideAllPapers();
    clearFilterButtonStates();
  });
}

function filterPapersByType(type: string, relatedPapers: RelatedPaper[]): void {
  relatedPapers.forEach((paper, index) => {
    const card = document.getElementById(`related-card-${index}`);
    if (!card) return;

    const shouldShow = getRelatedPaperFilterType(paper) === type;

    card.style.display = shouldShow ? "block" : "none";
  });
}

function showAllPapers(): void {
  const cards = document.querySelectorAll('[id^="related-card-"]');
  cards.forEach(card => {
    (card as HTMLElement).style.display = "block";
  });
}

function hideAllPapers(): void {
  const cards = document.querySelectorAll('[id^="related-card-"]');
  cards.forEach(card => {
    (card as HTMLElement).style.display = "none";
  });
}

function updateFilterButtonStates(activeButton: HTMLElement): void {
  const filterButtons = document.querySelectorAll(".filter-chip");
  filterButtons.forEach(button => {
    button.classList.remove("ring-2", "ring-blue-500");
  });
  activeButton.classList.add("ring-2", "ring-blue-500");
}

function clearFilterButtonStates(): void {
  const filterButtons = document.querySelectorAll(".filter-chip");
  filterButtons.forEach(button => {
    button.classList.remove("ring-2", "ring-blue-500");
  });
}

function loadAndDisplayPaper(): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    try {
      const urlParams = new URLSearchParams(window.location.search);
      const paperId = urlParams.get("id");

      if (!paperId) {
        throw new Error("No paper ID provided");
      }

      // Try to load from multi-perspective cache
      const cacheKey = `paper-cache-multi-${paperId}`;
      const cachedData = localStorage.getItem(cacheKey);

      if (!cachedData) {
        throw new Error("Multi-perspective paper data not found in cache");
      }

      let graphResult: GraphResultMulti;
      try {
        const parsedData = JSON.parse(cachedData) as unknown;
        // Validate cached data against current schema
        graphResult = GraphResultMultiSchema.parse(parsedData);
        console.log(`[Detail-Multi] Cache validation successful for paper ${paperId}`);
      } catch (schemaError) {
        // Schema validation failed - cache is outdated
        console.log(
          `[Detail-Multi] INVALID SCHEMA! Paper ${paperId} cache outdated, removing from cache...`,
        );
        console.warn(`[Detail-Multi] Schema validation error:`, schemaError);
        localStorage.removeItem(cacheKey);
        throw new Error(
          `Multi-perspective paper data is outdated. Please re-evaluate the paper to generate new results.`,
        );
      }

      // Hide loading state
      const loading = document.getElementById("loading-detail");
      const content = document.getElementById("paper-content");
      if (loading) loading.style.display = "none";
      if (content) content.classList.remove("hidden");

      // Display paper information
      displayPaperInfo(graphResult);

      // Display multi-perspective evaluation
      displayMultiPerspectiveEvaluation(graphResult.paper.evaluation);

      // Display structured analysis if available
      const structuredAnalysisContainer =
        document.getElementById("structured-analysis");
      const structuredAnalysisSection = document.querySelector(
        '[data-section="structured-analysis"]',
      );
      if (structuredAnalysisContainer && structuredAnalysisSection) {
        if (graphResult.paper.evaluation.structured) {
          structuredAnalysisContainer.innerHTML = createStructuredAnalysisDisplay(
            graphResult.paper.evaluation.structured,
            graphResult,
          );
          setupSectionToggle("structured-analysis");
        } else {
          // Hide the entire section if no structured analysis exists
          structuredAnalysisSection.classList.add("hidden");
        }
      }

      // Display graph
      createHierarchicalGraph(graphResult.graph, "#graph-svg", (entity: Entity) => {
        if (entity.excerpts && entity.excerpts.length > 0) {
          showExcerptsModal(entity);
        }
      });

      // Display paper terms
      const paperTermsContainer = document.getElementById("paper-terms");
      if ((graphResult.background || graphResult.target) && paperTermsContainer) {
        paperTermsContainer.innerHTML = createPaperTermsDisplay(
          graphResult.background ?? null,
          graphResult.target ?? null,
          null,
        );
        paperTermsContainer.classList.remove("hidden");
      }

      // Display related papers
      displayRelatedPapers(graphResult.related, graphResult.background);

      // Set up section toggles
      setupSectionToggle("multi-perspective-evaluation");
      setupSectionToggle("paper-graph");
      setupSectionToggle("paper-terms");
      setupSectionToggle("related-papers");

      resolve();
    } catch (error) {
      console.error("Error loading paper:", error);

      const loading = document.getElementById("loading-detail");
      const errorDiv = document.getElementById("error-detail");

      if (loading) loading.style.display = "none";
      if (errorDiv) errorDiv.classList.remove("hidden");

      reject(error instanceof Error ? error : new Error(String(error)));
    }
  });
}

function displayPaperInfo(graphResult: GraphResultMulti): void {
  const paper = graphResult.paper;

  // Title and authors
  const titleEl = document.getElementById("paper-title");
  const authorsEl = document.getElementById("paper-authors");
  if (titleEl) titleEl.innerHTML = renderLatex(paper.title);
  if (authorsEl) authorsEl.textContent = paper.authors.join(", ");

  // Basic info
  const yearEl = document.getElementById("paper-year");
  const conferenceEl = document.getElementById("paper-conference");
  const approvalEl = document.getElementById("paper-approval");
  const ratingEl = document.getElementById("paper-rating");
  const arxivEl = document.getElementById("paper-arxiv");

  if (yearEl) yearEl.textContent = paper.year.toString();
  if (conferenceEl) conferenceEl.textContent = formatConferenceName(paper.conference);

  if (approvalEl) {
    const approvalText =
      paper.approval === true
        ? "Accepted"
        : paper.approval === false
          ? "Rejected"
          : "Unknown";
    const approvalColor =
      paper.approval === true
        ? "text-green-600 dark:text-green-400"
        : paper.approval === false
          ? "text-red-600 dark:text-red-400"
          : "text-gray-600 dark:text-gray-400";
    approvalEl.className = `text-lg font-semibold ${approvalColor}`;
    approvalEl.textContent = approvalText;
  }

  // Use overall assessment confidence for novelty score
  if (ratingEl) {
    const scoreOutOfFive = Math.round(paper.evaluation.probability * 5 * 10) / 10;
    ratingEl.textContent = `${scoreOutOfFive}/5`;
  }

  if (paper.arxiv_id && arxivEl) {
    arxivEl.innerHTML = `<a href="${getArxivUrl(paper.arxiv_id)}" target="_blank" rel="noopener noreferrer" class="text-blue-600 dark:text-blue-400 hover:underline">${paper.arxiv_id}</a>`;
  } else if (arxivEl) {
    arxivEl.textContent = "N/A";
  }

  // Abstract
  const abstractEl = document.getElementById("paper-abstract");
  if (abstractEl) abstractEl.innerHTML = renderLatex(paper.abstract);

  // Keywords (extract from graph entities)
  const keywordsEl = document.getElementById("paper-keywords");
  if (keywordsEl) {
    const keywords = graphResult.graph.entities
      .filter(entity => entity.type === "keyword")
      .map(entity => entity.label)
      .slice(0, 8);

    keywordsEl.innerHTML = keywords
      .map(
        keyword =>
          `<span class="px-2 py-1 bg-teal-100/70 dark:bg-teal-900/30 text-teal-800 dark:text-teal-300 text-xs rounded-md border border-teal-300/50 dark:border-teal-700/50">${cleanKeyword(keyword)}</span>`,
      )
      .join(" ");
  }
}

/**
 * Setup event delegation for evidence expansion buttons
 */
function setupEvidenceExpansionHandlers(): void {
  document.addEventListener("click", event => {
    const target = event.target as HTMLElement;
    if (
      target.classList.contains("evidence-expand-btn") ||
      target.closest(".evidence-expand-btn")
    ) {
      event.preventDefault();

      const button = target.classList.contains("evidence-expand-btn")
        ? target
        : target.closest(".evidence-expand-btn");

      if (!button) return;

      const evidenceIndex = (button as HTMLElement).dataset.evidenceIndex;
      if (!evidenceIndex) return;

      const detailsElement = document.getElementById(
        `evidence-details-${evidenceIndex}`,
      );
      const expandIcon = button.querySelector(".expand-icon");
      const expandText = button.querySelector(".expand-text");

      if (detailsElement && expandIcon && expandText) {
        const isHidden = detailsElement.classList.contains("hidden");

        // Toggle visibility
        detailsElement.classList.toggle("hidden", !isHidden);

        // Update icon rotation
        expandIcon.classList.toggle("rotate-180", isHidden);

        // Update button text
        expandText.textContent = isHidden ? "Hide details" : "Show details";
      }
    }
  });
}

/**
 * Setup event delegation for related paper links
 */
function setupRelatedPaperLinkHandlers(): void {
  document.addEventListener("click", event => {
    const target = event.target as HTMLElement;
    if (target.classList.contains("related-paper-link")) {
      event.preventDefault();

      // Scroll to Related Papers section
      const relatedPapersHeader = document.getElementById("related-papers-header");
      if (relatedPapersHeader) {
        relatedPapersHeader.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    }
  });
}

async function initializeDetailMultiPage(): Promise<void> {
  // Multi-perspective feature is currently disabled - redirect to main search
  // To re-enable: remove this redirect and uncomment the toggle in search.html
  window.location.href = buildPageUrl("search.html");
  return;

  await waitForDOM();

  if (isMobileDevice()) {
    showMobileMessage();
    return;
  }

  // Setup event delegation for related paper links
  setupRelatedPaperLinkHandlers();

  // Setup event delegation for evidence expansion
  setupEvidenceExpansionHandlers();

  await retryWithBackoff(() => loadAndDisplayPaper());
  addFooter();
}

// Initialize the application
initializeDetailMultiPage().catch(showInitError);
