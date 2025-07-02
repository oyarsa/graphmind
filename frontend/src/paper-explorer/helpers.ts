import katex from "katex";
import "katex/dist/katex.min.css";
import { RelatedPaper } from "./model";

/**
 * Formats a scientific citation from paper information.
 * @param authors - Array of author names (full names)
 * @param year - Publication year
 * @param title - Paper title (used as fallback)
 * @returns Formatted citation string
 */
export function formatScientificCitation(
  authors: string[] | null | undefined,
  year: number | null | undefined,
  title: string,
): string {
  // Extract last names from authors
  const getLastName = (fullName: string): string => {
    const parts = fullName.trim().split(/\s+/);
    return parts[parts.length - 1];
  };

  if (authors && authors.length > 0 && year) {
    const firstAuthorLastName = getLastName(authors[0]);
    if (authors.length === 1) {
      return `${firstAuthorLastName} (${year})`;
    } else {
      return `${firstAuthorLastName} et al. (${year})`;
    }
  } else if (authors && authors.length > 0) {
    const firstAuthorLastName = getLastName(authors[0]);
    if (authors.length === 1) {
      return firstAuthorLastName;
    } else {
      return `${firstAuthorLastName} et al.`;
    }
  } else if (year) {
    return `(${year})`;
  } else {
    return title;
  }
}

/**
 * Sets up a collapsible section with header, content, and chevron toggle.
 * @param baseId - The base ID for the section (e.g., "paper-terms", "related-papers")
 */
export function setupSectionToggle(baseId: string): void {
  const header = document.getElementById(`${baseId}-header`);
  const content = document.getElementById(baseId);
  const chevron = document.getElementById(`${baseId}-chevron`);

  if (header && content && chevron) {
    // Open by default - remove hidden class and rotate chevron
    content.classList.remove("hidden");
    chevron.classList.add("rotate-180");

    header.addEventListener("click", () => {
      const isHidden = content.classList.contains("hidden");
      content.classList.toggle("hidden", !isHidden);
      chevron.classList.toggle("rotate-180", isHidden);
    });
  }
}

/**
 * Creates HTML display for paper terms including background, target, and primary area
 */
export function createPaperTermsDisplay(
  background: string | null,
  target: string | null,
  primaryArea: string | null = null,
): string {
  if (!background && !target && !primaryArea) {
    return `<div class="text-gray-600 dark:text-gray-500 text-sm">
      No analysis data available
    </div>`;
  }

  let html = "";

  // Background, Target, and Primary Area section
  if (background || target || primaryArea) {
    html += `<div class="space-y-3">`;

    if (primaryArea) {
      html += `
        <div class="rounded-lg border border-gray-200 bg-gray-50/50 p-4
                    dark:border-gray-700 dark:bg-gray-800/50">
          <div class="mb-2 flex items-center gap-2">
            <div class="h-4 w-1 rounded-full bg-green-500"></div>
            <h4 class="text-sm font-semibold tracking-wide text-gray-900 uppercase
                       dark:text-gray-100">
              Primary Area
            </h4>
          </div>
          <p class="text-sm leading-relaxed text-gray-700 dark:text-gray-300">
            ${primaryArea}
          </p>
        </div>`;
    }

    if (background) {
      html += `
        <div class="rounded-lg border border-gray-200 bg-gray-50/50 p-4
                    dark:border-gray-700 dark:bg-gray-800/50">
          <div class="mb-2 flex items-center gap-2">
            <div class="h-4 w-1 rounded-full bg-blue-500"></div>
            <h4 class="text-sm font-semibold tracking-wide text-gray-900 uppercase
                       dark:text-gray-100">
              Background
            </h4>
          </div>
          <p class="text-sm leading-relaxed text-gray-700 dark:text-gray-300">
            ${background}
          </p>
        </div>`;
    }

    if (target) {
      html += `
        <div class="rounded-lg border border-gray-200 bg-gray-50/50 p-4
                    dark:border-gray-700 dark:bg-gray-800/50">
          <div class="mb-2 flex items-center gap-2">
            <div class="h-4 w-1 rounded-full bg-purple-500"></div>
            <h4 class="text-sm font-semibold tracking-wide text-gray-900 uppercase
                       dark:text-gray-100">
              Target
            </h4>
          </div>
          <p class="text-sm leading-relaxed text-gray-700 dark:text-gray-300">
            ${target}
          </p>
        </div>`;
    }

    html += `</div>`;
  }

  return html;
}

/**
 * Determine relationship type and styling for a related paper
 */
export function getRelationshipStyle(paper: RelatedPaper) {
  if (paper.source === "semantic" && paper.polarity === "positive") {
    return {
      type: "target",
      label: "Target",
      icon: "🧠",
      color: "bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300",
      style: "rounded-full", // Rounded for semantic
    };
  } else if (paper.source === "semantic" && paper.polarity === "negative") {
    return {
      type: "background",
      label: "Background",
      icon: "🧠",
      color: "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300",
      style: "rounded-full", // Rounded for semantic
    };
  } else if (paper.source === "citations" && paper.polarity === "positive") {
    return {
      type: "supporting",
      label: "Supporting",
      icon: "🔗",
      color:
        "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-300",
      style: "rounded-md", // Square for citations
    };
  } else {
    return {
      type: "contrasting",
      label: "Contrasting",
      icon: "🔗",
      color: "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300",
      style: "rounded-md", // Square for citations
    };
  }
}

/**
 * Calculate score percentage and color based on score value
 */
export function getScoreDisplay(score: number) {
  const scorePercent = Math.round(score * 100);
  const scoreColor =
    score >= 0.7 ? "bg-green-500" : score >= 0.4 ? "bg-yellow-500" : "bg-red-500";

  return { scorePercent, scoreColor };
}

/**
 * Format type names for display (capitalize and replace underscores)
 */
export function formatTypeName(type: string): string {
  const formatted = type.replace(/_/g, " ");
  return formatted.charAt(0).toUpperCase() + formatted.slice(1);
}

/**
 * Transform conference names to proper case
 */
export function formatConferenceName(
  conference: string | null | undefined,
): string | null {
  if (!conference) return null;

  const transformations: Record<string, string> = {
    iclr: "ICLR",
    neurips: "NeurIPS",
  };

  return transformations[conference.toLowerCase()] || conference;
}

/**
 * Generate arXiv URL from arXiv ID, if available.
 */
export function getArxivUrl(arxivId?: string | null): string | null {
  if (!arxivId) return null;
  return `https://arxiv.org/abs/${arxivId}`;
}

/**
 * Format paper title for display as citation link.
 * Uses "Author et al. (Year)" format when metadata is available,
 * otherwise falls back to first 20 characters of title.
 */
export function formatPaperCitation(paper: RelatedPaper): string {
  if (paper.authors && paper.authors.length > 0 && paper.year) {
    const firstAuthor = paper.authors[0];
    // Extract only the last name from the first author
    const lastName = firstAuthor.split(" ").pop() ?? firstAuthor;
    const suffix = paper.authors.length > 1 ? " et al." : "";
    return `${lastName}${suffix} (${paper.year})`;
  }

  // Fallback to first 20 characters of title
  return paper.title.length > 20 ? `${paper.title.substring(0, 20)}...` : paper.title;
}

/**
 * Create a reusable side-by-side comparison component for background/target analysis.
 *
 * @param leftTitle - Title for the left column (e.g., "Main Paper")
 * @param leftContent - Content for the left column
 * @param rightTitle - Title for the right column (e.g., "Related Paper")
 * @param rightContent - Content for the right column
 * @param sectionType - Type of comparison ("background" or "target") for color theming
 * @returns HTML string for the side-by-side comparison
 */
export function createSideBySideComparison(
  leftTitle: string,
  leftContent: string | null,
  rightTitle: string,
  rightContent: string | null,
  sectionType: "background" | "target",
): string {
  if (!leftContent && !rightContent) {
    return "";
  }

  const formatTitle = sectionType === "background" ? "Background" : "Target";
  const notchColor = sectionType === "background" ? "bg-blue-500" : "bg-purple-500";

  return `
    <div class="mb-4">
      <div class="mb-3 flex items-center gap-2">
        <div class="h-4 w-1 rounded-full ${notchColor}"></div>
        <h5 class="text-sm font-semibold tracking-wide text-gray-900 uppercase dark:text-gray-100">
          ${formatTitle} Comparison
        </h5>
      </div>

      <div class="grid gap-4 md:grid-cols-2">
        <!-- Left Column -->
        <div class="space-y-2">
          <h6 class="text-xs font-medium text-gray-600 dark:text-gray-400 uppercase tracking-wide">
            ${leftTitle}
          </h6>
          <div class="text-sm leading-relaxed text-gray-700 dark:text-gray-300">
            ${leftContent ? renderLatex(leftContent) : '<span class="text-gray-500 dark:text-gray-500 italic">No data available</span>'}
          </div>
        </div>

        <!-- Right Column -->
        <div class="space-y-2">
          <h6 class="text-xs font-medium text-gray-600 dark:text-gray-400 uppercase tracking-wide">
            ${rightTitle}
          </h6>
          <div class="text-sm leading-relaxed text-gray-700 dark:text-gray-300">
            ${rightContent ? renderLatex(rightContent) : '<span class="text-gray-500 dark:text-gray-500 italic">No data available</span>'}
          </div>
        </div>
      </div>
    </div>
  `;
}

// Import types - we need to define a minimal interface to avoid circular imports
interface EvidenceItem {
  text: string;
  paper_title?: string | null;
  paper_id?: string | null;
  source?: "citations" | "semantic" | null;
}

interface RelatedPaperForEvidence {
  source: "citations" | "semantic";
  polarity: "positive" | "negative";
  background?: string | null;
  target?: string | null;
  contexts?:
    | {
        sentence: string;
        polarity: "positive" | "negative" | null;
      }[]
    | null;
  title: string;
  authors?: string[] | null;
  year?: number | null;
}

/**
 * Create an expandable evidence item with comparison information.
 *
 * @param evidence - The evidence item (string or EvidenceItem)
 * @param evidenceIndex - Unique index for this evidence item
 * @param relatedPaper - The related paper object if found
 * @param relatedPaperIndex - Index in related papers array for linking
 * @param mainPaperBackground - Main paper's background text
 * @param mainPaperTarget - Main paper's target text
 * @param bulletColor - Color class for the bullet point
 * @returns HTML string for the expandable evidence item
 */
export function createExpandableEvidenceItem(
  evidence: string | EvidenceItem,
  evidenceIndex: string,
  relatedPaper: RelatedPaperForEvidence | null,
  relatedPaperIndex: number | null,
  mainPaperBackground: string | null,
  mainPaperTarget: string | null,
  bulletColor: string,
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
  if (evidence.paper_title) {
    // Format display text as scientific citation when available, otherwise use title
    const displayText = relatedPaper
      ? formatScientificCitation(
          relatedPaper.authors,
          relatedPaper.year,
          evidence.paper_title,
        )
      : evidence.paper_title;

    const paperTitleElement =
      relatedPaperIndex !== null
        ? `<a href="#related-papers"
             class="related-paper-link hover:underline cursor-pointer text-blue-800 dark:text-blue-200"
             data-paper-index="${relatedPaperIndex}">
             ${renderLatex(displayText)}:</a>`
        : `<a href="#related-papers"
             class="related-paper-link hover:underline cursor-pointer text-blue-800 dark:text-blue-200">
             ${renderLatex(displayText)}:</a>`;

    const hasSemanticContent =
      relatedPaper?.source === "semantic" &&
      (relatedPaper.background != null || relatedPaper.target != null);

    const hasCitationContent =
      relatedPaper?.source === "citations" &&
      relatedPaper.contexts &&
      relatedPaper.contexts.length > 0;

    const hasExpandableContent = hasSemanticContent || hasCitationContent;

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
                data-evidence-index="${evidenceIndex}"
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
            <div class="evidence-details hidden mt-3 pl-4 border-l-2 border-gray-200 dark:border-gray-700" id="evidence-details-${evidenceIndex}">
              ${createEvidenceComparisonContent(relatedPaper, mainPaperBackground, mainPaperTarget)}
            </div>
          `
              : ""
          }
        </div>
      </li>
    `;
  }

  // Fallback for evidence with just text
  return `
    <li class="flex items-start gap-2">
      <span class="mt-1.5 block h-1.5 w-1.5 flex-shrink-0 rounded-full ${bulletColor}"></span>
      <span class="text-sm leading-relaxed text-gray-700 dark:text-gray-300">
        ${renderLatex(evidence.text)}
      </span>
    </li>
  `;
}

/**
 * Create comparison content for evidence details based on paper type.
 */
function createEvidenceComparisonContent(
  relatedPaper: RelatedPaperForEvidence,
  mainPaperBackground: string | null,
  mainPaperTarget: string | null,
): string {
  if (relatedPaper.source === "semantic") {
    // For semantic papers, show background/target comparison
    if (
      relatedPaper.polarity === "negative" &&
      (relatedPaper.background ?? mainPaperBackground)
    ) {
      return createSideBySideComparison(
        "Main Paper",
        mainPaperBackground,
        "Related Paper",
        relatedPaper.background ?? null,
        "background",
      );
    } else if (
      relatedPaper.polarity === "positive" &&
      (relatedPaper.target ?? mainPaperTarget)
    ) {
      return createSideBySideComparison(
        "Main Paper",
        mainPaperTarget,
        "Related Paper",
        relatedPaper.target ?? null,
        "target",
      );
    }
  } else if (relatedPaper.contexts) {
    // For citation papers, show citation contexts
    return `
      <div class="mb-4">
        <div class="mb-3 flex items-center gap-2">
          <h6 class="text-sm font-semibold tracking-wide text-gray-900 uppercase dark:text-gray-100">
            Citation Contexts
          </h6>
        </div>
        <div class="space-y-2">
          ${relatedPaper.contexts
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
                ${renderLatex(context.sentence)}
              </span>
            </div>
          `,
            )
            .join("")}
        </div>
      </div>
    `;
  }

  return "";
}

/**
 * Render any LaTeX fragments in the input text to KaTeX HTML.
 *
 * • Display maths: wrap in $$ … $$ (may span multiple lines, but must be non-empty)
 * • Inline  maths: wrap in  $ … $  (must stay on a single line)
 * • All non-math text is HTML-escaped to prevent XSS.
 */
export function renderLatex(text: string): string {
  if (!text) return "";

  const escapeHtml = (str: string): string => {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  };

  const math_pattern = /(\$\$[\s\S]+?\$\$|\$[^\n$]+\$)/g;

  return text
    .split(math_pattern)
    .map(segment => {
      const display = segment.startsWith("$$") && segment.endsWith("$$");
      const inline = !display && segment.startsWith("$") && segment.endsWith("$");

      if (display || inline) {
        const body = display ? segment.slice(2, -2) : segment.slice(1, -1);
        try {
          return katex.renderToString(body, {
            displayMode: display,
            throwOnError: false,
            errorColor: "#cc0000",
            output: "html",
          });
        } catch (err) {
          console.error("KaTeX error:", err);
          return escapeHtml(segment); // fall back to raw, safely escaped
        }
      }

      return escapeHtml(segment); // plain text
    })
    .join("");
}
