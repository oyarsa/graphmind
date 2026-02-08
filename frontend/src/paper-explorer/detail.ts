import { z } from "zod/v4";
import { retryWithBackoff, waitForDOM, showInitError, cleanKeyword } from "../util";
import {
  GraphResult,
  GraphResultSchema,
  RelatedPaper,
  Graph,
  Entity,
  StructuredEval,
} from "./model";
import {
  getRelationshipStyle,
  getScoreDisplay,
  setupSectionToggle,
  renderLatex,
  renderLatexWithCitationFootnotes,
  citationReferenceFromRelatedPaper,
  buildCitationIndexFromReferences,
  RenderLatexOptions,
  getArxivUrl,
  formatConferenceName,
  createExpandableEvidenceItem,
  getScoreColor,
  createHierarchicalGraph,
  normalizePaperTitle,
  findRelatedPaperIndexByTitle,
} from "./helpers";
import { addFooter } from "../footer";
import { parseStoredJson } from "./storage";
import { loadSettings } from "./settings";
import { buildDetailPageContext, PaperChatWidget } from "./chat";

let detailChatWidget: PaperChatWidget | null = null;

function createRelatedPaperCard(
  paper: RelatedPaper,
  index: number,
  latexOptions: RenderLatexOptions,
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
          title="Go up to Evaluation section"
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

      ${
        paper.contexts && paper.contexts.length > 0
          ? `
      <!-- Citation Contexts Section -->
      <div class="mb-4">
        <div class="mb-3 flex items-center gap-2">
          <div class="h-4 w-1 rounded-full bg-purple-500"></div>
          <h5
            class="text-sm font-semibold tracking-wide text-gray-900 uppercase
                   dark:text-gray-100"
          >
            Citation Contexts
          </h5>
        </div>
        <div class="space-y-2">
          ${paper.contexts
            .map(
              context => `
                <div class="flex items-start gap-2">
                  ${
                    context.polarity
                      ? `<span class="mt-0.5 inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${
                          context.polarity === "positive"
                            ? "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300"
                            : "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300"
                        }">
                          ${context.polarity === "positive" ? "+" : "-"}
                        </span>`
                      : ""
                  }
                  <span class="text-sm leading-relaxed text-gray-700 dark:text-gray-300 flex-1">
                    ${renderLatexWithCitationFootnotes(context.sentence, {
                      ...latexOptions,
                      fallbackReference: citationReferenceFromRelatedPaper(paper),
                    })}
                  </span>
                </div>
              `,
            )
            .join("")}
        </div>
      </div>
      `
          : ""
      }

      ${
        paper.source === "semantic" && paper.polarity === "positive" && paper.target
          ? `
      <div class="mb-4">
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
      </div>
      `
          : ""
      }

      ${
        paper.source === "semantic" && paper.polarity === "negative" && paper.background
          ? `
      <div class="mb-4">
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
      </div>
      `
          : ""
      }
    </div>
  </div>
   `;
}

/**
 * Filter related papers to only include those that appear in the evidence lists.
 *
 * This ensures the Related Papers section only shows papers that are actually
 * referenced in the supporting or contradictory evidence sections.
 */
function filterToEvidencePapers(
  relatedPapers: RelatedPaper[],
  evaluation: StructuredEval | null | undefined,
): RelatedPaper[] {
  if (!evaluation) return relatedPapers;

  // Collect all paper titles from evidence lists
  const evidenceTitles = new Set<string>();

  for (const evidence of evaluation.supporting_evidence) {
    if (typeof evidence === "object" && evidence.paper_title) {
      evidenceTitles.add(normalizePaperTitle(evidence.paper_title));
    }
  }

  for (const evidence of evaluation.contradictory_evidence) {
    if (typeof evidence === "object" && evidence.paper_title) {
      evidenceTitles.add(normalizePaperTitle(evidence.paper_title));
    }
  }

  // Filter related papers to only those in evidence
  return relatedPapers.filter(paper =>
    evidenceTitles.has(normalizePaperTitle(paper.title)),
  );
}

function createStructuredEvaluationDisplay(
  evaluation: StructuredEval,
  evidencePapers: RelatedPaper[],
  latexOptions: RenderLatexOptions,
): string {
  // Note: renderEvidence function moved to createExpandableEvidenceItem helper

  // Calculate rating display if probability is available
  let ratingBadge = "";
  if (evaluation.probability != null) {
    const probability = evaluation.probability;
    const rating = Math.min(5, Math.max(1, Math.ceil(probability * 5)));
    const ratingDisplay = `${rating}/5`;
    const ratingColor =
      rating <= 2
        ? "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300 border-red-300 dark:border-red-700"
        : rating === 3
          ? "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300 border-yellow-300 dark:border-yellow-700"
          : "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300 border-green-300 dark:border-green-700";
    ratingBadge = `<span class="px-2 py-0.5 text-xs font-medium rounded-md border ${ratingColor}">
      Novelty: ${ratingDisplay}
    </span>`;
  }

  return `
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
          ${ratingBadge}
        </div>
        <p class="text-sm leading-relaxed text-gray-700 dark:text-gray-300">
          ${renderLatex(evaluation.conclusion)}
        </p>
      </div>

      <!-- Paper Summary -->
      ${
        evaluation.paper_summary
          ? `
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
      `
          : ""
      }

      <!-- Key Comparisons -->
      ${
        evaluation.key_comparisons.length > 0
          ? `
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
      `
          : ""
      }

      <!-- Supporting Evidence -->
      ${
        evaluation.supporting_evidence.length > 0
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
            ${evaluation.supporting_evidence
              .map((evidence, index) => {
                const relatedPaperIndex =
                  typeof evidence === "object" && evidence.paper_title
                    ? findRelatedPaperIndexByTitle(evidence.paper_title, evidencePapers)
                    : null;
                const relatedPaper =
                  relatedPaperIndex !== null ? evidencePapers[relatedPaperIndex] : null;

                return createExpandableEvidenceItem(
                  evidence,
                  `supporting-${index}`,
                  relatedPaper,
                  relatedPaperIndex,
                  "bg-green-500",
                  latexOptions,
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
        evaluation.contradictory_evidence.length > 0
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
            ${evaluation.contradictory_evidence
              .map((evidence, index) => {
                const relatedPaperIndex =
                  typeof evidence === "object" && evidence.paper_title
                    ? findRelatedPaperIndexByTitle(evidence.paper_title, evidencePapers)
                    : null;
                const relatedPaper =
                  relatedPaperIndex !== null ? evidencePapers[relatedPaperIndex] : null;

                return createExpandableEvidenceItem(
                  evidence,
                  `contradictory-${index}`,
                  relatedPaper,
                  relatedPaperIndex,
                  "bg-red-500",
                  latexOptions,
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

function createHierarchicalGraphWrapper(graph: Graph): void {
  const container = document.getElementById("paper-graph");
  if (!container) return;

  // Use the shared hierarchical graph function with node click handler
  createHierarchicalGraph(graph, "#graph-svg", (entity: Entity) => {
    if (entity.excerpts && entity.excerpts.length > 0) {
      showExcerptsModal(entity);
    }
  });
}

/**
 * Render related papers
 */
function renderRelatedPapers(
  relatedPapers: RelatedPaper[],
  latexOptions: RenderLatexOptions,
): void {
  const relatedPapersContainer = document.getElementById("related-papers-content");
  if (!relatedPapersContainer) return;

  if (relatedPapers.length === 0) {
    relatedPapersContainer.innerHTML = `
      <div class="text-gray-600 dark:text-gray-500 text-sm text-center py-4">
        No related papers available.
      </div>
    `;
    return;
  }

  // Render all papers
  relatedPapersContainer.innerHTML = relatedPapers
    .map((relatedPaper, index) =>
      createRelatedPaperCard(relatedPaper, index, latexOptions),
    )
    .join("");

  // Add click event listeners for expansion
  relatedPapers.forEach((_, index) => {
    const cardHeader = document.querySelector(`#related-card-${index} .card-header`);
    const expandedContent = document.getElementById(`expanded-content-${index}`);
    const goBackBtn = cardHeader?.querySelector(".go-back-btn");

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
        updateGoBackButtonVisibility(goBackBtn as HTMLElement | null, !isExpanded);
      });

      // Handle go back button click
      if (goBackBtn) {
        goBackBtn.addEventListener("click", event => {
          event.stopPropagation(); // Prevent card expansion
          scrollToEvaluationSection();
        });
      }

      // Set initial go back button visibility
      const isInitiallyExpanded = !expandedContent.classList.contains("hidden");
      updateGoBackButtonVisibility(
        goBackBtn as HTMLElement | null,
        isInitiallyExpanded,
      );
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
 * Scroll to the Evaluation section
 */
function scrollToEvaluationSection(): void {
  const evaluationHeader = document.getElementById("structured-evaluation-header");

  if (evaluationHeader) {
    evaluationHeader.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

/**
 * Expand the Related Papers section if it's collapsed.
 * Returns true if the section was expanded, false if it was already expanded.
 */
function ensureRelatedPapersSectionExpanded(): boolean {
  const content = document.getElementById("related-papers");
  const chevron = document.getElementById("related-papers-chevron");

  if (content?.classList.contains("hidden")) {
    // Section is collapsed, expand it
    content.classList.remove("hidden");
    chevron?.classList.add("rotate-180");
    return true;
  }
  return false;
}

/**
 * Function to expand a related paper card by index and scroll to it
 */
function expandRelatedPaper(index: number): void {
  // First ensure the Related Papers section is expanded
  const wasCollapsed = ensureRelatedPapersSectionExpanded();

  const card = document.getElementById(`related-card-${index}`);
  const expandedContent = document.getElementById(`expanded-content-${index}`);
  const goBackBtn = card?.querySelector(".go-back-btn") as HTMLElement | null;

  if (card && expandedContent) {
    // Expand the card if it's not already expanded
    if (expandedContent.classList.contains("hidden")) {
      expandedContent.classList.remove("hidden");
      card.querySelector(".expand-icon")?.classList.add("rotate-180");

      // Show the go back button when expanded
      updateGoBackButtonVisibility(goBackBtn, true);
    }

    // If the section was just expanded, wait for layout to settle before scrolling
    if (wasCollapsed) {
      requestAnimationFrame(() => {
        card.scrollIntoView({ behavior: "smooth", block: "center" });
      });
    } else {
      // Scroll to the card with some offset for better visibility
      card.scrollIntoView({ behavior: "smooth", block: "center" });
    }
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

      const paperIndex = target.dataset.paperIndex;
      if (paperIndex) {
        // Expand specific paper if index is available
        const index = parseInt(paperIndex, 10);
        expandRelatedPaper(index);
      } else {
        // Expand section if collapsed, then scroll to Related Papers section
        const wasCollapsed = ensureRelatedPapersSectionExpanded();
        const relatedPapersHeader = document.getElementById("related-papers-header");
        if (relatedPapersHeader) {
          if (wasCollapsed) {
            requestAnimationFrame(() => {
              relatedPapersHeader.scrollIntoView({
                behavior: "smooth",
                block: "start",
              });
            });
          } else {
            relatedPapersHeader.scrollIntoView({ behavior: "smooth", block: "start" });
          }
        }
      }
    }
  });
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
          <div class="text-sm leading-relaxed text-gray-700 dark:text-gray-300">
            ${renderLatex(excerpt.text)}
          </div>
        </div>
      `,
      )
      .join("");
  }

  // Show modal
  modal.style.display = "flex";

  // Close modal handlers
  const closeModal = () => (modal.style.display = "none");
  closeButton.onclick = closeModal;

  // Click outside to close
  modal.onclick = event => {
    if (event.target === modal) {
      closeModal();
    }
  };

  // Escape key to close
  const escapeHandler = (event: KeyboardEvent) => {
    if (event.key === "Escape") {
      closeModal();
      document.removeEventListener("keydown", escapeHandler);
    }
  };
  document.addEventListener("keydown", escapeHandler);
}

function loadPaperDetail(): void {
  const urlParams = new URLSearchParams(window.location.search);
  const paperId = urlParams.get("id");

  const loadingEl = document.getElementById("loading-detail");
  const contentEl = document.getElementById("paper-content");
  const errorEl = document.getElementById("error-detail");

  try {
    // Check if we have a paper ID
    if (!paperId) {
      throw new Error("No paper ID provided");
    }

    // Try to find the paper from multiple sources
    let graphResult: GraphResult | null = null;

    // First, check if it's an individual paper (arXiv or cached)
    const individualPaper = localStorage.getItem(`paper-cache-${paperId}`);
    if (individualPaper) {
      console.log(`[Detail] Loading paper ${paperId} from individual cache`);
      try {
        const parsedData = JSON.parse(individualPaper) as unknown;
        // Validate cached data against current schema
        graphResult = GraphResultSchema.parse(parsedData);
        console.log(`[Detail] Cache validation successful for paper ${paperId}`);
      } catch (schemaError) {
        // Schema validation failed - cache is outdated
        console.log(
          `[Detail] INVALID SCHEMA! Paper ${paperId} cache outdated, removing from cache...`,
        );
        console.warn(`[Detail] Schema validation error:`, schemaError);
        localStorage.removeItem(`paper-cache-${paperId}`);
        graphResult = null; // Will fall through to JSON dataset check
      }
    }

    if (!graphResult) {
      // Second, check the JSON dataset
      console.log(
        `[Detail] Paper ${paperId} not in individual cache, checking JSON dataset`,
      );
      const storedPapers = localStorage.getItem("papers-dataset");
      if (storedPapers) {
        const graphResults = parseStoredJson(
          storedPapers,
          z.array(GraphResultSchema),
          "papers-dataset",
        );
        if (graphResults) {
          graphResult = graphResults.find(gr => gr.paper.id === paperId) ?? null;
          if (graphResult) {
            console.log(`[Detail] Found paper ${paperId} in JSON dataset`);
          } else {
            console.log(`[Detail] Paper ${paperId} not found in JSON dataset either`);
          }
        } else {
          localStorage.removeItem("papers-dataset");
        }
      }
    }

    if (!graphResult) {
      throw new Error("Paper not found");
    }

    const paper = graphResult.paper;
    const citationIndex = buildCitationIndexFromReferences(paper.references);
    const latexOptions: RenderLatexOptions = { citationIndex };

    document.title = `${paper.title} - Paper Explorer`;

    // Populate the content
    const titleEl = document.getElementById("paper-title");
    const authorsEl = document.getElementById("paper-authors");
    const yearEl = document.getElementById("paper-year");
    const conferenceEl = document.getElementById("paper-conference");
    const abstractEl = document.getElementById("paper-abstract");
    const approvalEl = document.getElementById("paper-approval");
    const ratingEl = document.getElementById("paper-rating");
    const arxivEl = document.getElementById("paper-arxiv");

    if (titleEl) titleEl.innerHTML = renderLatex(paper.title);
    if (authorsEl) authorsEl.textContent = paper.authors.join(", ");
    if (yearEl) yearEl.textContent = paper.year.toString();
    if (conferenceEl) conferenceEl.textContent = formatConferenceName(paper.conference);
    if (abstractEl) {
      abstractEl.innerHTML = renderLatexWithCitationFootnotes(
        paper.abstract,
        latexOptions,
      );
    }
    // Handle Decision field - hide entire container when approval is null
    const decisionContainer = document.getElementById("decision-container");
    if (approvalEl && decisionContainer) {
      if (paper.approval === null) {
        // Hide the entire Decision container for arXiv papers
        decisionContainer.classList.add("hidden");
      } else {
        // Show the container and populate with approval status
        decisionContainer.classList.remove("hidden");
        const approvalText = paper.approval ? "Approved" : "Rejected";
        const approvalClass = paper.approval
          ? "text-green-600 dark:text-green-400 font-semibold"
          : "text-red-600 dark:text-red-400 font-semibold";

        approvalEl.textContent = approvalText;
        approvalEl.className = approvalClass;
      }
    }
    if (ratingEl) {
      // Check if structured evaluation has probability
      if (paper.structured_evaluation?.probability != null) {
        const probability = paper.structured_evaluation.probability;
        // Convert probability (0-1) to rating (1-5)
        const rating = Math.min(5, Math.max(1, Math.ceil(probability * 5)));
        const novelText = `${rating}/5`;
        // Color based on rating: 1-2 = red, 3 = yellow, 4-5 = green
        const novelClass =
          rating <= 2
            ? "text-red-600 dark:text-red-400 font-semibold"
            : rating === 3
              ? "text-yellow-600 dark:text-yellow-400 font-semibold"
              : "text-green-600 dark:text-green-400 font-semibold";

        ratingEl.textContent = novelText;
        ratingEl.className = novelClass;
      } else {
        // Fallback to original High/Low display
        const isNovel = paper.rating >= 3;
        const novelText = isNovel ? "High" : "Low";
        const novelClass = isNovel
          ? "text-green-600 dark:text-green-400 font-semibold"
          : "text-red-600 dark:text-red-400 font-semibold";

        ratingEl.textContent = novelText;
        ratingEl.className = novelClass;
      }
    }
    if (arxivEl) {
      const arxivUrl = getArxivUrl(paper.arxiv_id);
      if (arxivUrl) {
        arxivEl.innerHTML = `
          <a href="${arxivUrl}" target="_blank" rel="noopener noreferrer"
             class="text-blue-600 dark:text-blue-400 hover:text-blue-500
                    dark:hover:text-blue-300 underline transition-colors duration-200">
            ${paper.arxiv_id}
          </a>
        `;
      } else {
        arxivEl.textContent = "Not available";
        arxivEl.className = "text-lg font-semibold text-gray-400 dark:text-gray-600";
      }
    }

    // Handle keywords from graph entities
    const keywordsContainer = document.getElementById("paper-keywords");
    if (keywordsContainer) {
      const keywords = graphResult.graph.entities
        .filter(e => e.type === "keyword")
        .map(e => cleanKeyword(e.label))
        .slice(0, 5);

      if (keywords.length > 0) {
        keywordsContainer.innerHTML = keywords
          .map(
            keyword => `
            <span class="px-2 py-1 bg-teal-100/70 dark:bg-teal-900/30 text-teal-800
                         dark:text-teal-300 text-xs rounded-md border border-teal-300/50
                         dark:border-teal-700/50">
              ${keyword}
            </span>
          `,
          )
          .join("");
      } else {
        keywordsContainer.innerHTML = `
          <span class="text-gray-600 dark:text-gray-500 text-sm">
            No keywords available
          </span>
        `;
      }
    }

    // Filter related papers to only those that appear in evidence lists
    // This ensures consistency between evidence section and Related Papers section
    const evidencePapers = filterToEvidencePapers(
      graphResult.related,
      paper.structured_evaluation,
    );

    // Handle structured evaluation
    const structuredEvalContainer = document.getElementById("structured-evaluation");
    const structuredEvalSection = document.querySelector(
      '[data-section="structured-evaluation"]',
    );
    if (structuredEvalContainer && structuredEvalSection) {
      if (paper.structured_evaluation) {
        structuredEvalContainer.innerHTML = createStructuredEvaluationDisplay(
          paper.structured_evaluation,
          evidencePapers,
          latexOptions,
        );
        setupSectionToggle("structured-evaluation");
      } else {
        // Hide the entire section if no structured evaluation exists
        structuredEvalSection.classList.add("hidden");
      }
    }

    setupSectionToggle("paper-graph");
    createHierarchicalGraphWrapper(graphResult.graph);

    // Handle related papers - only show papers that appear in evidence lists
    const relatedPapersContentContainer = document.getElementById(
      "related-papers-content",
    );
    if (relatedPapersContentContainer) {
      if (evidencePapers.length > 0) {
        renderRelatedPapers(evidencePapers, latexOptions);
      } else {
        relatedPapersContentContainer.innerHTML = `
          <div class="text-gray-600 dark:text-gray-500 text-sm">
            No related papers available
          </div>
        `;
      }
    }

    // Setup Related Papers section toggle
    setupSectionToggle("related-papers");

    // Show content and hide loading
    if (loadingEl) loadingEl.style.display = "none";
    if (contentEl) contentEl.classList.remove("hidden");

    const pageContext = buildDetailPageContext(graphResult);
    if (detailChatWidget) {
      detailChatWidget.updateContext(() => pageContext);
    } else {
      const apiUrl = (import.meta.env.VITE_API_URL as string | undefined) ?? "";
      detailChatWidget = new PaperChatWidget({
        baseUrl: apiUrl,
        pageType: "detail",
        getPageContext: () => pageContext,
        getModel: () => loadSettings().llm_model,
      });
    }
  } catch (error) {
    console.error("Error loading paper details:", error);
    if (loadingEl) loadingEl.style.display = "none";
    if (errorEl) errorEl.classList.remove("hidden");
  }
}

/**
 * Reset all collapsible sections to their default expanded state.
 * This is called when the page is restored from bfcache to ensure
 * sections don't retain their collapsed state from previous visits.
 */
function resetSectionStates(): void {
  const sections = ["paper-graph", "structured-evaluation", "related-papers"];

  for (const baseId of sections) {
    const content = document.getElementById(baseId);
    const chevron = document.getElementById(`${baseId}-chevron`);

    if (content && chevron) {
      // Expand section (default state)
      content.classList.remove("hidden");
      chevron.classList.add("rotate-180");
    }
  }
}

/**
 * Initialise the Paper Detail application.
 */
async function initialiseApp(): Promise<void> {
  await waitForDOM();

  // Setup event delegation for related paper links
  setupRelatedPaperLinkHandlers();

  // Setup event delegation for evidence expansion
  setupEvidenceExpansionHandlers();

  // Handle bfcache restoration - reset sections to expanded state
  window.addEventListener("pageshow", event => {
    if (event.persisted) {
      resetSectionStates();
    }
  });

  await retryWithBackoff(() => {
    loadPaperDetail();
    return Promise.resolve();
  });

  // Add footer
  addFooter();
}

initialiseApp().catch(showInitError);
