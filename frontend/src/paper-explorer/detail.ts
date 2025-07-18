import * as d3 from "d3";
import {
  retryWithBackoff,
  waitForDOM,
  showInitError,
  isMobileDevice,
  showMobileMessage,
  cleanKeyword,
} from "../util";
import {
  GraphResult,
  GraphResultSchema,
  RelatedPaper,
  Graph,
  Entity,
  StructuredEval,
} from "./model";
import {
  createPaperTermsDisplay,
  getRelationshipStyle,
  getScoreDisplay,
  formatTypeName,
  setupSectionToggle,
  renderLatex,
  getArxivUrl,
  formatConferenceName,
  createSideBySideComparison,
  createExpandableEvidenceItem,
  getScoreColor,
} from "./helpers";
import { addFooter } from "../footer";

function createRelatedPaperCard(
  paper: RelatedPaper,
  index: number,
  mainPaperBackground?: string | null,
  mainPaperTarget?: string | null,
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
                   font-medium whitespace-nowrap"
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
                ${renderLatex(context.sentence)}
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
        paper.source === "semantic" &&
        paper.polarity === "positive" &&
        (paper.target || mainPaperTarget)
          ? createSideBySideComparison(
              "Main Paper",
              mainPaperTarget ?? null,
              "Related Paper",
              paper.target ?? null,
              "target",
            )
          : ""
      }

      ${
        paper.source === "semantic" &&
        paper.polarity === "negative" &&
        (paper.background || mainPaperBackground)
          ? createSideBySideComparison(
              "Main Paper",
              mainPaperBackground ?? null,
              "Related Paper",
              paper.background ?? null,
              "background",
            )
          : ""
      }
    </div>
  </div>
   `;
}

/**
 * Helper function to find the index of a related paper by title in the currently filtered papers
 */
/**
 * Normalize a paper title for matching by removing non-alphabetical characters
 * and converting to lowercase. This helps match titles that differ by punctuation.
 */
function normalizeTitle(title: string): string {
  return title
    .replace(/[^a-zA-Z0-9\s]/g, "")
    .toLowerCase()
    .trim();
}

function findRelatedPaperIndex(
  paperTitle: string,
  relatedPapers: RelatedPaper[],
  activeFilters: Set<string>,
): number | null {
  // Filter papers based on their relationship type (same logic as renderFilteredRelatedPapers)
  const filteredPapers = relatedPapers.filter(paper => {
    const relationship = getRelationshipStyle(paper);
    return activeFilters.has(relationship.type);
  });

  // Normalize the search title for comparison
  const normalizedSearchTitle = normalizeTitle(paperTitle);

  // Find the index in the filtered array using normalized title matching
  const index = filteredPapers.findIndex(
    paper => normalizeTitle(paper.title) === normalizedSearchTitle,
  );
  return index >= 0 ? index : null;
}

function createStructuredEvaluationDisplay(
  evaluation: StructuredEval,
  graphResult: GraphResult | null,
  activeFilters: Set<string>,
  mainPaperBackground?: string | null,
  mainPaperTarget?: string | null,
): string {
  // Note: renderEvidence function moved to createExpandableEvidenceItem helper

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
        </div>
        <p class="text-sm leading-relaxed text-gray-700 dark:text-gray-300">
          ${renderLatex(evaluation.conclusion)}
        </p>
      </div>

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
                  typeof evidence === "object" && evidence.paper_title && graphResult
                    ? findRelatedPaperIndex(
                        evidence.paper_title,
                        graphResult.related,
                        activeFilters,
                      )
                    : null;
                const relatedPaper =
                  relatedPaperIndex !== null && graphResult
                    ? graphResult.related[relatedPaperIndex]
                    : null;

                return createExpandableEvidenceItem(
                  evidence,
                  `supporting-${index}`,
                  relatedPaper,
                  relatedPaperIndex,
                  mainPaperBackground ?? null,
                  mainPaperTarget ?? null,
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
                  typeof evidence === "object" && evidence.paper_title && graphResult
                    ? findRelatedPaperIndex(
                        evidence.paper_title,
                        graphResult.related,
                        activeFilters,
                      )
                    : null;
                const relatedPaper =
                  relatedPaperIndex !== null && graphResult
                    ? graphResult.related[relatedPaperIndex]
                    : null;

                return createExpandableEvidenceItem(
                  evidence,
                  `contradictory-${index}`,
                  relatedPaper,
                  relatedPaperIndex,
                  mainPaperBackground ?? null,
                  mainPaperTarget ?? null,
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

function createHierarchicalGraph(graph: Graph): void {
  const container = document.getElementById("paper-graph");
  if (!container) return;

  const svg = d3.select("#graph-svg");
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
      if (d.excerpts && d.excerpts.length > 0) {
        event.stopPropagation();
        showExcerptsModal(d);
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
      .attr("y", totalHeight - 20)
      .attr("text-anchor", "middle")
      .attr("font-size", "13px")
      .attr("fill", "#6b7280")
      .attr("class", "dark:fill-gray-500")
      .text("Click on nodes to view additional details");
  }
}

/**
 * Setup filtering functionality for related papers
 */
function setupRelatedPapersFiltering(
  relatedPapers: RelatedPaper[],
  mainPaperBackground?: string | null,
  mainPaperTarget?: string | null,
): void {
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
  const countRelation = (src: string, pol: string) =>
    relatedPapers.filter(p => p.source === src && p.polarity === pol).length;
  const counts = {
    background: countRelation("semantic", "negative"),
    target: countRelation("semantic", "positive"),
    supporting: countRelation("citations", "positive"),
    contrasting: countRelation("citations", "negative"),
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
      renderFilteredRelatedPapers(
        relatedPapers,
        activeFilters,
        mainPaperBackground,
        mainPaperTarget,
      );
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
      renderFilteredRelatedPapers(
        relatedPapers,
        activeFilters,
        mainPaperBackground,
        mainPaperTarget,
      );
    });
  }

  // Add click handler for hide all button
  if (hideAllButton) {
    hideAllButton.addEventListener("click", () => {
      activeFilters.clear();

      updateChipStates();
      renderFilteredRelatedPapers(
        relatedPapers,
        activeFilters,
        mainPaperBackground,
        mainPaperTarget,
      );
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
  mainPaperBackground?: string | null,
  mainPaperTarget?: string | null,
): void {
  const relatedPapersContainer = document.getElementById("related-papers-content");
  if (!relatedPapersContainer) return;

  // Filter papers based on their relationship type
  const filteredPapers = relatedPapers.filter(paper => {
    const relationship = getRelationshipStyle(paper);
    return activeFilters.has(relationship.type);
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
    .map((relatedPaper, index) =>
      createRelatedPaperCard(relatedPaper, index, mainPaperBackground, mainPaperTarget),
    )
    .join("");

  // Add click event listeners for expansion
  filteredPapers.forEach((_, index) => {
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
 * Function to expand a related paper card by index and scroll to it
 */
function expandRelatedPaper(index: number): void {
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

    // Scroll to the card with some offset for better visibility
    card.scrollIntoView({ behavior: "smooth", block: "center" });
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

      // First, enable all paper types to ensure the target paper is visible
      const showAllButton = document.getElementById(
        "filter-show-all",
      ) as HTMLButtonElement | null;
      if (showAllButton) {
        showAllButton.click();
      }

      const paperIndex = target.dataset.paperIndex;
      if (paperIndex) {
        // Expand specific paper if index is available
        const index = parseInt(paperIndex, 10);
        expandRelatedPaper(index);
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
        const graphResults = JSON.parse(storedPapers) as GraphResult[];
        graphResult = graphResults.find(gr => gr.paper.id === paperId) ?? null;
        if (graphResult) {
          console.log(`[Detail] Found paper ${paperId} in JSON dataset`);
        } else {
          console.log(`[Detail] Paper ${paperId} not found in JSON dataset either`);
        }
      }
    }

    if (!graphResult) {
      throw new Error("Paper not found");
    }

    const paper = graphResult.paper;

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
    if (abstractEl) abstractEl.innerHTML = renderLatex(paper.abstract);
    if (approvalEl) {
      const approvalText =
        paper.approval === null ? "?" : paper.approval ? "Approved" : "Rejected";
      const approvalClass =
        paper.approval === null
          ? "text-gray-600 dark:text-gray-400 font-semibold cursor-help inline-flex items-center justify-center w-5 h-5 border border-gray-400 dark:border-gray-500 rounded-full text-xs"
          : paper.approval
            ? "text-green-600 dark:text-green-400 font-semibold"
            : "text-red-600 dark:text-red-400 font-semibold";

      approvalEl.textContent = approvalText;
      approvalEl.className = approvalClass;

      // Only add tooltip and immediate hover for null approval
      if (paper.approval === null) {
        // Remove any existing event listeners to avoid duplicates
        approvalEl.onmouseenter = null;
        approvalEl.onmouseleave = null;

        // Create immediate tooltip effect
        let tooltipDiv: HTMLElement | null = null;

        approvalEl.onmouseenter = () => {
          // Create custom tooltip for immediate display
          tooltipDiv = document.createElement("div");
          tooltipDiv.textContent =
            "Approval decision is not available for arXiv papers";
          tooltipDiv.className =
            "absolute z-50 bg-gray-900 text-white text-xs rounded px-2 py-1 pointer-events-none whitespace-nowrap";
          tooltipDiv.style.bottom = "100%";
          tooltipDiv.style.left = "50%";
          tooltipDiv.style.transform = "translateX(-50%)";
          tooltipDiv.style.marginBottom = "4px";

          approvalEl.style.position = "relative";
          approvalEl.appendChild(tooltipDiv);
        };

        approvalEl.onmouseleave = () => {
          if (tooltipDiv) {
            tooltipDiv.remove();
            tooltipDiv = null;
          }
        };
      } else {
        // Clear everything for defined approval values
        approvalEl.onmouseenter = null;
        approvalEl.onmouseleave = null;
      }

      // Always clear the title attribute to prevent native tooltip
      approvalEl.title = "";
    }
    if (ratingEl) {
      // Check if structured evaluation has probability
      if (paper.structured_evaluation?.probability != null) {
        const probability = paper.structured_evaluation.probability;
        const probabilityPercent = Math.round(probability * 100);
        const novelText = `${probabilityPercent}%`;
        const novelClass =
          probability >= 0.5
            ? "text-green-600 dark:text-green-400 font-semibold"
            : "text-red-600 dark:text-red-400 font-semibold";

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

    // Handle paper terms
    const paperTermsContainer = document.getElementById("paper-terms");
    if (paperTermsContainer) {
      // Extract primary area from graph entities
      const primaryAreaEntity = graphResult.graph.entities.find(
        e => e.type === "primary_area",
      );
      const primaryAreaText = primaryAreaEntity?.label ?? null;

      paperTermsContainer.innerHTML = createPaperTermsDisplay(
        graphResult.background ?? null,
        graphResult.target ?? null,
        primaryAreaText,
      );
    }

    // Handle structured evaluation
    const structuredEvalContainer = document.getElementById("structured-evaluation");
    const structuredEvalSection = document.querySelector(
      '[data-section="structured-evaluation"]',
    );
    if (structuredEvalContainer && structuredEvalSection) {
      if (paper.structured_evaluation) {
        // Use default active filters (all types visible by default)
        const defaultActiveFilters = new Set([
          "background",
          "target",
          "supporting",
          "contrasting",
        ]);
        structuredEvalContainer.innerHTML = createStructuredEvaluationDisplay(
          paper.structured_evaluation,
          graphResult,
          defaultActiveFilters,
          graphResult.background,
          graphResult.target,
        );
        setupSectionToggle("structured-evaluation");
      } else {
        // Hide the entire section if no structured evaluation exists
        structuredEvalSection.classList.add("hidden");
      }
    }

    setupSectionToggle("paper-terms");
    setupSectionToggle("paper-graph");
    createHierarchicalGraph(graphResult.graph);

    // Handle related papers
    const relatedPapersContentContainer = document.getElementById(
      "related-papers-content",
    );
    if (relatedPapersContentContainer) {
      if (graphResult.related.length > 0) {
        // Setup filtering functionality
        setupRelatedPapersFiltering(
          graphResult.related,
          graphResult.background,
          graphResult.target,
        );

        // Initial render with all papers visible
        renderFilteredRelatedPapers(
          graphResult.related,
          new Set(["background", "target", "supporting", "contrasting"]),
          graphResult.background,
          graphResult.target,
        );
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
  } catch (error) {
    console.error("Error loading paper details:", error);
    if (loadingEl) loadingEl.style.display = "none";
    if (errorEl) errorEl.classList.remove("hidden");
  }
}

/**
 * Initialise the Paper Detail application.
 */
async function initialiseApp(): Promise<void> {
  await waitForDOM();

  if (isMobileDevice()) {
    showMobileMessage();
    return;
  }

  // Setup event delegation for related paper links
  setupRelatedPaperLinkHandlers();

  // Setup event delegation for evidence expansion
  setupEvidenceExpansionHandlers();

  await retryWithBackoff(() => {
    loadPaperDetail();
    return Promise.resolve();
  });

  // Add footer
  addFooter();
}

initialiseApp().catch(showInitError);
