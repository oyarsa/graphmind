import * as d3 from "d3";
import katex from "katex";
import "katex/dist/katex.min.css";
import { RelatedPaper, Graph, Entity } from "./model";

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
            ${stripLatexCitations(primaryArea)}
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
            ${stripLatexCitations(background)}
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
            ${stripLatexCitations(target)}
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
 * Get color for score progress bar (red to green gradient)
 */
export function getScoreColor(score: number): string {
  // Convert to percentage for color calculation
  const percent = score * 100;

  if (percent < 50) {
    // Red to yellow gradient (0-50%)
    const ratio = percent / 50;
    const r = 239; // red-500
    const g = Math.round(68 + (245 - 68) * ratio); // transition to yellow-400
    const b = 68;
    return `rgb(${r}, ${g}, ${b})`;
  } else {
    // Yellow to green gradient (50-100%)
    const ratio = (percent - 50) / 50;
    const r = Math.round(245 - (245 - 34) * ratio); // transition from yellow-400 to green-600
    const g = Math.round(158 + (197 - 158) * ratio);
    const b = Math.round(11 + (77 - 11) * ratio);
    return `rgb(${r}, ${g}, ${b})`;
  }
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
            ${leftContent ? renderLatex(stripLatexCitations(leftContent)) : '<span class="text-gray-500 dark:text-gray-500 italic">No data available</span>'}
          </div>
        </div>

        <!-- Right Column -->
        <div class="space-y-2">
          <h6 class="text-xs font-medium text-gray-600 dark:text-gray-400 uppercase tracking-wide">
            ${rightTitle}
          </h6>
          <div class="text-sm leading-relaxed text-gray-700 dark:text-gray-300">
            ${rightContent ? renderLatex(stripLatexCitations(rightContent)) : '<span class="text-gray-500 dark:text-gray-500 italic">No data available</span>'}
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
              ${createEvidenceComparisonContent(relatedPaper)}
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
): string {
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
            ${renderLatex(stripLatexCitations(relatedPaper.background))}
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
            ${renderLatex(stripLatexCitations(relatedPaper.target))}
          </p>
        </div>
      `;
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
                ${renderLatex(stripLatexCitations(context.sentence))}
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
 * Strips raw LaTeX commands from text for display.
 *
 * Removes:
 * - Citation commands: ~\citep{...}, \cite{...}, \citep{...}, \citet{...}, etc.
 * - Malformed citations with space: ~\ cite{...}, \ citep{...}
 * - Footnotes: \footnote{...}, ~\ footnote{...}
 * - Other LaTeX commands: \textbf{...}, \emph{...}, etc.
 * - Standalone tildes (non-breaking space in LaTeX)
 *
 * @param text - Text potentially containing LaTeX commands
 * @returns Cleaned text suitable for display
 */
export function stripLatexCitations(text: string): string {
  if (!text) return "";

  return (
    text
      // Remove footnotes (including malformed with space after backslash)
      .replace(/~?\s*\\\s*footnote\s*\{[^}]*\}/gi, "")
      // Remove citation commands (including malformed with space after backslash)
      // Handles: ~\citep{...}, \cite{...}, \ cite{...}, ~\ citep{...}, etc.
      .replace(/~?\s*\\\s*cite[pt]?\s*\{[^}]*\}/gi, "")
      .replace(/~?\s*\\\s*citealp\s*\{[^}]*\}/gi, "")
      .replace(/~?\s*\\\s*citealt\s*\{[^}]*\}/gi, "")
      .replace(/~?\s*\\\s*citeauthor\s*\{[^}]*\}/gi, "")
      .replace(/~?\s*\\\s*citeyear\s*\{[^}]*\}/gi, "")
      .replace(/~?\s*\\\s*parencite\s*\{[^}]*\}/gi, "")
      .replace(/~?\s*\\\s*textcite\s*\{[^}]*\}/gi, "")
      // Remove other common LaTeX commands with arguments
      .replace(/\\\s*[a-zA-Z]+\s*(\[[^\]]*\])?\s*\{[^}]*\}/g, "")
      // Remove standalone tildes (non-breaking space)
      .replace(/~/g, " ")
      // Clean up any double spaces left behind
      .replace(/\s+/g, " ")
      .trim()
  );
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

/**
 * Creates a hierarchical graph visualization for a paper's structure.
 * This function renders entities and their relationships in a layered format.
 *
 * @param graph - The graph data containing entities and relationships
 * @param svgSelector - CSS selector for the SVG element (default: "#graph-svg")
 * @param onNodeClick - Optional callback for node click events
 */
export function createHierarchicalGraph(
  graph: Graph,
  svgSelector = "#graph-svg",
  onNodeClick?: (entity: Entity, event: MouseEvent) => void,
): void {
  const svg = d3.select(svgSelector);
  svg.selectAll("*").remove();

  const nodeHeight = 85;
  const titleNodeWidth = 280; // Wider for title
  const nodeWidth = 200;
  const levelHeight = 110;
  const nodeSpacing = 45; // Space between nodes
  const padding = 60; // Reduced padding on sides

  // Group entities by type in hierarchical order (remove keywords, tldr, primary_area)
  const levelTypes = [["title"], ["claim"], ["method"], ["experiment"]];

  const levels = levelTypes.map(types =>
    graph.entities.filter(entity => types.includes(entity.type)),
  );

  // Filter out empty levels
  const nonEmptyLevels = levels.filter(level => level.length > 0);

  // Calculate minimum required width based on content, with reasonable bounds
  const maxNodesInLevel = Math.max(...nonEmptyLevels.map(level => level.length));
  const minRequiredWidth = Math.max(
    titleNodeWidth + 2 * padding, // Minimum for title
    maxNodesInLevel * (nodeWidth + nodeSpacing) - nodeSpacing + 2 * padding, // Width for widest level
  );

  // Use calculated width with minimum of 800px and maximum of 1200px
  const graphWidth = Math.max(800, Math.min(minRequiredWidth, 1200));

  // Calculate positions with centered layout
  const allNodes: (Entity & { x: number; y: number; level: number; width: number })[] =
    [];
  const nodesByLabel = new Map<
    string,
    Entity & { x: number; y: number; level: number; width: number }
  >();

  nonEmptyLevels.forEach((level, levelIndex) => {
    const levelY = levelIndex * levelHeight + 60;

    if (levelIndex === 0) {
      // Title - center it and make it wider
      const titleNode = {
        ...level[0],
        x: graphWidth / 2,
        y: levelY,
        level: levelIndex,
        width: titleNodeWidth,
      };
      allNodes.push(titleNode);
      nodesByLabel.set(titleNode.label, titleNode);
    } else {
      // For other levels, distribute nodes evenly with consistent spacing
      const totalContentWidth =
        level.length * nodeWidth + (level.length - 1) * nodeSpacing;
      const startX = (graphWidth - totalContentWidth) / 2;

      level.forEach((entity, idx) => {
        const node = {
          ...entity,
          x: startX + nodeWidth / 2 + idx * (nodeWidth + nodeSpacing),
          y: levelY,
          level: levelIndex,
          width: nodeWidth,
        };
        allNodes.push(node);
        nodesByLabel.set(node.label, node);
      });
    }
  });

  // Create edges based on relationships, with special handling for removed nodes
  const edges = graph.relationships
    .map(rel => {
      const source = nodesByLabel.get(rel.source);
      const target = nodesByLabel.get(rel.target);

      // If both nodes exist in our filtered graph, create the edge
      if (source && target) {
        return { source, target };
      }

      // Handle cases where intermediate nodes (tldr, primary_area) were removed
      // If source is title and target is removed node, find what the removed node connects to
      const titleNode = [...nodesByLabel.values()].find(n => n.type === "title");
      const claimNodes = [...nodesByLabel.values()].filter(n => n.type === "claim");

      // If we have a title and claims, but no direct connections, create them
      if (
        titleNode &&
        claimNodes.length > 0 &&
        rel.source === titleNode.label &&
        ["primary_area", "tldr"].some(type =>
          graph.entities.find(e => e.type === type && e.label === rel.target),
        )
      ) {
        // Find what this removed node connects to
        const downstreamRels = graph.relationships.filter(r => r.source === rel.target);
        return downstreamRels
          .map(downRel => {
            const downTarget = nodesByLabel.get(downRel.target);
            return downTarget ? { source: titleNode, target: downTarget } : null;
          })
          .filter(Boolean);
      }

      return null;
    })
    .flat()
    .filter(edge => edge !== null);

  // Ensure title connects to claims if no connections exist
  const titleNode = [...nodesByLabel.values()].find(n => n.type === "title");
  const claimNodes = [...nodesByLabel.values()].filter(n => n.type === "claim");

  if (titleNode && claimNodes.length > 0) {
    const hasDirectConnections = edges.some(
      edge => edge.source.type === "title" || edge.target.type === "title",
    );

    if (!hasDirectConnections) {
      // Create direct connections from title to all claims
      claimNodes.forEach(claimNode => {
        edges.push({ source: titleNode, target: claimNode });
      });
    }
  }

  // Color scheme for different node types (Transformer-inspired colors)
  const nodeColors: Record<string, string> = {
    title: "#A8D5BA", // soft sage green (like attention mechanisms)
    primary_area: "#FFE4E1", // light peach (like embeddings)
    tldr: "#FFF8DC", // cream (like Add & Norm layers)
    claim: "#FFE6CC", // slightly darker coral/peach
    method: "#CCE7FF", // darker blue (like Feed Forward layers)
    experiment: "#F0E6FF", // light purple (like Linear layers)
  };

  // Calculate total height and set SVG dimensions
  const totalHeight = nonEmptyLevels.length * levelHeight + 120;

  // Set explicit SVG dimensions to override CSS classes
  svg
    .attr("width", graphWidth)
    .attr("height", totalHeight)
    .attr("viewBox", `0 0 ${graphWidth} ${totalHeight}`)
    .style("max-width", "100%") // Ensure it doesn't overflow container
    .style("height", "auto")
    .style("display", "block")
    .style("margin", "0 auto"); // Center the SVG in its container

  // Draw edges with curves
  const edgeGroup = svg.append("g").attr("class", "edges");

  edgeGroup
    .selectAll("path")
    .data(edges)
    .join("path")
    .attr("d", d => {
      const sourceY = d.source.y + nodeHeight;
      const targetY = d.target.y;
      const midY = (sourceY + targetY) / 2;
      return `M ${d.source.x} ${sourceY}
              C ${d.source.x} ${midY},
                ${d.target.x} ${midY},
                ${d.target.x} ${targetY}`;
    })
    .attr("fill", "none")
    .attr("stroke", "#9ca3af")
    .attr("stroke-width", 2)
    .attr("opacity", 0.5)
    .attr(
      "class",
      "transition-all duration-200 hover:stroke-blue-500 hover:opacity-100",
    );

  // Draw nodes
  const nodeGroup = svg.append("g").attr("class", "nodes");

  const nodeSelection = nodeGroup
    .selectAll("g")
    .data(allNodes)
    .join("g")
    .attr("transform", d => `translate(${d.x - d.width / 2}, ${d.y})`)
    .style("cursor", d => (d.excerpts && d.excerpts.length > 0 ? "pointer" : "default"))
    .on("click", function (event: MouseEvent, d) {
      if (onNodeClick) {
        event.stopPropagation();
        onNodeClick(d, event);
      }
    });

  // Add solid color rectangles for nodes
  nodeSelection
    .append("rect")
    .attr("width", d => d.width)
    .attr("height", nodeHeight)
    .attr("rx", 10)
    .attr("fill", d => nodeColors[d.type] || "#6b7280") // defaults to gray-500
    .attr("stroke", "rgba(255,255,255,0.1)")
    .attr("stroke-width", 0.5)
    .attr("class", "transition-all duration-300 ease-out")
    .style("filter", "none")
    .on("mouseenter", function () {
      d3.select(this).attr("stroke", "#666666").attr("stroke-width", 1.5);
    })
    .on("mouseleave", function () {
      d3.select(this).attr("stroke", "rgba(255,255,255,0.1)").attr("stroke-width", 0.5);
    });

  // Add text labels with proper truncation and centering
  nodeSelection
    .append("foreignObject")
    .attr("x", 0)
    .attr("y", 0)
    .attr("width", d => d.width)
    .attr("height", nodeHeight)
    .style("pointer-events", "none")
    .append("xhtml:div")
    .attr("class", "node-content")
    .style("width", d => `${d.width}px`)
    .style("height", `${nodeHeight}px`)
    .style("padding", "12px")
    .style("box-sizing", "border-box")
    .style("color", "#2C2C2C")
    .style("font-size", "15px")
    .style("font-weight", "500")
    .style("text-align", "center")
    .style("line-height", "1.3")
    .style("text-shadow", "none")
    .style("overflow", "hidden")
    .style("display", "flex")
    .style("align-items", "center")
    .style("justify-content", "center")
    .each(function (d) {
      if (!(this instanceof HTMLElement)) return;

      // Create inner container for text with truncation
      const innerDiv = document.createElement("div");
      innerDiv.style.maxHeight = "3.9em"; // 3 lines * 1.3 line-height
      innerDiv.style.overflow = "hidden";
      innerDiv.style.position = "relative";
      innerDiv.style.width = "100%";
      innerDiv.style.lineHeight = "1.3";
      innerDiv.style.fontSize = "15px";
      innerDiv.innerHTML = renderLatex(d.label);

      this.appendChild(innerDiv);
    });

  // Create legend at top-right (compact and narrower)
  const uniqueTypes = [...new Set(allNodes.map(n => n.type))];
  const legendGroup = svg.append("g").attr("class", "legend");
  const legendWidth = 104; // Increased by 30%
  const itemHeight = 21; // Increased by 30%

  // Fixed position in upper right corner
  const legendX = graphWidth - legendWidth - 20; // 20px from right edge
  const legendY = 20; // 20px from top

  legendGroup
    .append("rect")
    .attr("x", legendX - 10) // Increased padding proportionally
    .attr("y", legendY - 10) // Increased padding proportionally
    .attr("width", legendWidth + 13) // Increased padding proportionally
    .attr("height", uniqueTypes.length * itemHeight + 10) // Increased padding proportionally
    .attr("fill", "white")
    .attr("stroke", "#e5e7eb")
    .attr("rx", 4)
    .attr("opacity", 0.95)
    .attr("class", "dark:fill-gray-800 dark:stroke-gray-600");

  uniqueTypes.forEach((type, idx) => {
    const y = legendY + idx * itemHeight + 4;

    legendGroup
      .append("rect")
      .attr("x", legendX)
      .attr("y", y - 5)
      .attr("width", 10)
      .attr("height", 10)
      .attr("rx", 1)
      .attr("fill", nodeColors[type] || "#6b7280");

    legendGroup
      .append("text")
      .attr("x", legendX + 15)
      .attr("y", y)
      .attr("font-size", "14px")
      .attr("dominant-baseline", "middle")
      .attr("fill", "currentColor")
      .attr("class", "dark:fill-gray-300")
      .text(formatTypeName(type));
  });

  // Add help text if any nodes have excerpts
  const hasExcerpts = allNodes.some(n => n.excerpts && n.excerpts.length > 0);
  if (hasExcerpts) {
    svg
      .append("text")
      .attr("x", graphWidth / 2)
      .attr("y", totalHeight - 10)
      .attr("text-anchor", "middle")
      .attr("font-size", "13px")
      .attr("fill", "#6b7280")
      .attr("class", "dark:fill-gray-500")
      .text("Click on nodes to view additional details");
  }
}
