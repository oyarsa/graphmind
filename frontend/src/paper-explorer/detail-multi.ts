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
  GraphResultMulti,
  GraphResultMultiSchema,
  RelatedPaper,
  Graph,
  Entity,
  PerspectiveEvaluation,
  MultiEvaluationResult,
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
  formatTypeName,
} from "./helpers";
import { addFooter } from "../footer";

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
          <div class="${relationship.color} rounded-md px-2 py-1">
            <span class="text-xs font-medium">${relationship.icon} ${relationship.label}</span>
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
      .attr("y", totalHeight - 10)
      .attr("text-anchor", "middle")
      .attr("font-size", "13px")
      .attr("fill", "#6b7280")
      .attr("class", "dark:fill-gray-500")
      .text("Click on nodes to view additional details");
  }
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
  const backgroundPapers = relatedPapers.filter(p => p.background);
  const targetPapers = relatedPapers.filter(p => p.target);
  const supportingPapers = relatedPapers.filter(
    p => p.source === "citations" && p.polarity === "positive",
  );
  const contrastingPapers = relatedPapers.filter(
    p => p.source === "citations" && p.polarity === "negative",
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

    let shouldShow = false;

    switch (type) {
      case "background":
        shouldShow = !!paper.background;
        break;
      case "target":
        shouldShow = !!paper.target;
        break;
      case "supporting":
        shouldShow = paper.source === "citations" && paper.polarity === "positive";
        break;
      case "contrasting":
        shouldShow = paper.source === "citations" && paper.polarity === "negative";
        break;
    }

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

      const graphResult = GraphResultMultiSchema.parse(JSON.parse(cachedData));

      // Hide loading state
      const loading = document.getElementById("loading-detail");
      const content = document.getElementById("paper-content");
      if (loading) loading.style.display = "none";
      if (content) content.classList.remove("hidden");

      // Display paper information
      displayPaperInfo(graphResult);

      // Display multi-perspective evaluation
      displayMultiPerspectiveEvaluation(graphResult.paper.evaluation);

      // Display graph
      createHierarchicalGraph(graphResult.graph);

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

async function initializeDetailMultiPage(): Promise<void> {
  await waitForDOM();

  if (isMobileDevice()) {
    showMobileMessage();
    return;
  }

  await retryWithBackoff(() => loadAndDisplayPaper());
  addFooter();
}

// Initialize the application
initializeDetailMultiPage().catch(showInitError);
