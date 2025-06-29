import katex from "katex";
import "katex/dist/katex.min.css";
import { RelatedPaper } from "./model";

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
    .map((segment) => {
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
