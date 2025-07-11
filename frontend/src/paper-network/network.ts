import * as d3 from "d3";

import {
  DataService,
  getNodeId,
  getNodePos,
  type D3Link,
  type D3Node,
  type GraphLink,
  type LinkType,
  type PaperNeighbour,
  type RelatedPapersResponse,
  type SearchResult,
} from "./model";
import { requireElement, showDialog } from "../util";

interface PaperTypeConfig {
  pool: Map<string, PaperNeighbour[]>;
  shown: Map<string, number>;
  apiCall: (nodeId: string, limit: number) => Promise<RelatedPapersResponse>;
  linkType: LinkType;
}

export class PaperNetwork {
  private static readonly INITIAL_PAPERS_PER_TYPE = 2;

  private svg: d3.Selection<SVGSVGElement, unknown, HTMLElement, unknown>;
  private container: d3.Selection<SVGGElement, unknown, HTMLElement, unknown>;
  private linkGroup: d3.Selection<SVGGElement, unknown, HTMLElement, unknown>;
  private nodeGroup: d3.Selection<SVGGElement, unknown, HTMLElement, unknown>;
  private simulation: d3.Simulation<D3Node, D3Link>;
  private width = 0;
  private height = 0;
  private nodes: D3Node[] = [];
  private links: D3Link[] = [];
  private nodeMap = new Map<string, D3Node>();
  private expandedNodes = new Set<string>();
  private citedPools = new Map<string, PaperNeighbour[]>();
  private semanticPools = new Map<string, PaperNeighbour[]>();
  private citedShown = new Map<string, number>();
  private semanticShown = new Map<string, number>();
  private tooltip: HTMLElement;
  private currentSelectedNode: D3Node | null = null;
  private expandCitedButtons: HTMLButtonElement[] = [];
  private expandSemanticButtons: HTMLButtonElement[] = [];

  constructor(private dataService: DataService) {
    this.svg = d3.select("#graph");

    this.tooltip = requireElement("tooltip");

    requireElement("infoPanel"); // Validate element exists

    // ==== Set up SVG
    this.container = this.svg.append("g");

    // Add zoom behavior
    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on("zoom", (event: d3.D3ZoomEvent<SVGSVGElement, unknown>) => {
        this.container.attr("transform", event.transform.toString());
      });

    this.svg.call(zoom);

    // Create groups for links and nodes
    this.linkGroup = this.container.append("g").attr("class", "links");
    this.nodeGroup = this.container.append("g").attr("class", "nodes");

    // ==== Set up force simulation
    this.simulation = d3
      .forceSimulation<D3Node, D3Link>()
      .force(
        "link",
        d3
          .forceLink<D3Node, D3Link>()
          .id(d => d.id)
          .distance(150),
      )
      .force("charge", d3.forceManyBody<D3Node>().strength(-500))
      .force("center", d3.forceCenter())
      .force("collision", d3.forceCollide<D3Node>().radius(90));

    this.setupEventListeners();
    this.handleResize();

    window.addEventListener("resize", () => this.handleResize());
  }

  private setupEventListeners(): void {
    const searchPaperBtn = requireElement("searchPaper");
    const clearGraphBtn = requireElement("clearGraph");
    const clearSearchBtn = requireElement("clearSearch");

    // Citation expansion buttons
    this.expandCitedButtons = [
      requireElement("expandCited1") as HTMLButtonElement,
      requireElement("expandCited2") as HTMLButtonElement,
      requireElement("expandCited3") as HTMLButtonElement,
    ];

    // Semantic expansion buttons
    this.expandSemanticButtons = [
      requireElement("expandSemantic1") as HTMLButtonElement,
      requireElement("expandSemantic2") as HTMLButtonElement,
      requireElement("expandSemantic3") as HTMLButtonElement,
    ];

    const paperTitleInput = requireElement("paperTitle") as HTMLInputElement;

    searchPaperBtn.addEventListener("click", () => {
      const title = paperTitleInput.value.trim();
      if (title) {
        this.searchPapers(title).catch((error: unknown) => {
          console.error("Error searching papers:", error);
        });
      }
    });

    clearGraphBtn.addEventListener("click", () => this.clearGraph());

    clearSearchBtn.addEventListener("click", () => {
      paperTitleInput.value = "";
      clearSearchBtn.style.display = "none";
      this.hideSearchResults();
      paperTitleInput.focus();
    });

    paperTitleInput.addEventListener("input", () => {
      if (paperTitleInput.value.trim()) {
        clearSearchBtn.style.display = "block";
      } else {
        clearSearchBtn.style.display = "none";
        this.hideSearchResults();
      }
    });

    paperTitleInput.addEventListener("keypress", e => {
      if (e.key === "Enter") {
        searchPaperBtn.click();
      }
    });

    // Set up expand button event listeners
    this.setupExpandButtons([
      {
        buttons: this.expandCitedButtons,
        expandFn: this.expandCitations.bind(this),
      },
      {
        buttons: this.expandSemanticButtons,
        expandFn: this.expandSemantic.bind(this),
      },
    ]);
  }

  private setupExpandButtons(
    configs: {
      buttons: HTMLElement[];
      expandFn: (nodeId: string, count: number) => Promise<void>;
    }[],
  ): void {
    configs.forEach(({ buttons, expandFn }) => {
      buttons.forEach((button, index) => {
        const count = index + 1; // +1, +2, +3
        button.addEventListener("click", () => {
          if (this.currentSelectedNode) {
            void expandFn(this.currentSelectedNode.id, count);
          }
        });
      });
    });
  }

  private handleResize(): void {
    const container = requireElement("graphContainer");

    this.width = container.clientWidth;
    this.height = container.clientHeight;

    this.svg.attr("width", this.width).attr("height", this.height);

    this.simulation.force("center", d3.forceCenter(this.width / 2, this.height / 2));
  }

  private async searchPapers(query: string): Promise<void> {
    console.log("Starting search for:", query);
    this.showLoading(true, "Searching papers...");

    try {
      // Search for papers using data service
      const searchResults = await this.dataService.searchPapers(query, 20);
      console.log("Search completed, results:", searchResults.results.length);

      this.displaySearchResults(searchResults);
    } catch (error) {
      console.error("Error searching papers:", error);
      void showDialog("Error searching for papers. Please try again.", "Search Error");
    } finally {
      this.showLoading(false);
    }
  }

  private displaySearchResults(searchResults: SearchResult): void {
    console.log("Attempting to display search results...");

    let searchResultsDiv = document.getElementById("searchResults");
    let resultsCountDiv = document.getElementById("resultsCount");
    let resultsListDiv = document.getElementById("resultsList");

    console.log("Elements found:", {
      searchResultsDiv: !!searchResultsDiv,
      resultsCountDiv: !!resultsCountDiv,
      resultsListDiv: !!resultsListDiv,
    });

    // Create elements if they don't exist
    if (!searchResultsDiv || !resultsCountDiv || !resultsListDiv) {
      console.log("Creating missing search results elements...");
      this.createSearchResultsElements();

      // Re-query the elements after creation
      searchResultsDiv = document.getElementById("searchResults");
      resultsCountDiv = document.getElementById("resultsCount");
      resultsListDiv = document.getElementById("resultsList");

      if (!searchResultsDiv || !resultsCountDiv || !resultsListDiv) {
        console.error("Failed to create search results elements");
        return;
      }
    }

    console.log("Displaying", searchResults.results.length, "search results");

    if (searchResults.results.length === 0) {
      searchResultsDiv.style.display = "block";
      resultsCountDiv.textContent = "No results";
      resultsListDiv.innerHTML = `
        <div class="text-center py-8 px-5 text-gray-500 italic">
          No papers found. Try different search terms like "attention",
          "neural networks", or "machine learning".
       </div>
      `;
      return;
    }

    // Show results section
    searchResultsDiv.style.display = "block";
    resultsCountDiv.textContent = `${searchResults.results.length} results`;

    // Clear and populate results
    resultsListDiv.innerHTML = "";

    searchResults.results.forEach((result, index) => {
      console.log(`Creating result item ${index + 1}:`, result.title);

      const resultItem = document.createElement("div");
      resultItem.className =
        "bg-neutral-200/60 hover:bg-blue-100 dark:bg-neutral-800/60 dark:hover:bg-neutral-700" +
        " border border-neutral-400 hover:border-blue-500 dark:border-neutral-600" +
        " dark:hover:border-blue-400 rounded-lg p-3 mb-2 cursor-pointer transition-all duration-200";
      resultItem.setAttribute("data-paper-id", result.id);

      const authorsText =
        result.authors.length > 3
          ? result.authors.slice(0, 3).join(", ") + " et al."
          : result.authors.join(", ");

      resultItem.innerHTML = `
                <div class="text-sm font-semibold text-black dark:text-white mb-1.5
                            leading-snug">
                  ${result.title}
                </div>
                <div class="flex gap-4 items-center mb-2">
                    <div class="text-xs text-gray-700 dark:text-gray-300 flex-1">
                      ${authorsText}
                    </div>
                    <div class="text-xs text-gray-600 dark:text-gray-400 font-semibold">
                      ${result.year}
                    </div>
                    <div class="text-xs text-blue-600 dark:text-blue-500 bg-blue-500/15
                                px-1.5 py-0.5 rounded font-semibold">
                        ${Math.round(result.relevance * 100)}%
                    </div>
                </div>
            `;

      // Add click handler
      resultItem.addEventListener("click", () => {
        console.log("Result clicked:", result.title);
        void this.addPaperFromSearch(result.id);
        this.hideSearchResults();
      });

      resultsListDiv.appendChild(resultItem);
    });

    console.log("Search results display completed");
  }

  private createSearchResultsElements(): void {
    console.log("Creating search results elements...");
    const header = requireElement("header");

    // Check if search results already exist
    let searchResultsDiv = document.getElementById("searchResults");
    if (searchResultsDiv) {
      console.log("Search results div already exists");
      return;
    }

    // Create the search results structure
    searchResultsDiv = document.createElement("div");
    searchResultsDiv.id = "searchResults";
    searchResultsDiv.className =
      "max-h-[40vh] overflow-y-auto border-t border-neutral-300" +
      " dark:border-neutral-700 pt-4";
    searchResultsDiv.style.display = "none";

    const headerDiv = document.createElement("div");
    headerDiv.className = "mb-3 flex items-center justify-between px-1";

    const titleDiv = document.createElement("div");
    titleDiv.className = "text-sm font-semibold text-blue-600 dark:text-blue-500";
    titleDiv.textContent = "Search Results";

    const countDiv = document.createElement("div");
    countDiv.className = "text-xs text-gray-600 dark:text-gray-500";
    countDiv.id = "resultsCount";

    headerDiv.appendChild(titleDiv);
    headerDiv.appendChild(countDiv);

    const listDiv = document.createElement("div");
    listDiv.id = "resultsList";

    searchResultsDiv.appendChild(headerDiv);
    searchResultsDiv.appendChild(listDiv);

    header.appendChild(searchResultsDiv);

    console.log("Search results elements created successfully");
  }

  private hideSearchResults(): void {
    const searchResultsDiv = requireElement("searchResults");
    const paperTitleInput = requireElement("paperTitle") as HTMLInputElement;

    searchResultsDiv.style.display = "none";
    paperTitleInput.value = "";
  }

  private async addPaperFromSearch(paperId: string): Promise<void> {
    this.showLoading(true, "Adding paper to graph...");

    try {
      const paperDetails = await this.dataService.getPaperDetails(paperId);
      // The backend returned this paper as a search result, but somehow it doesn't
      // have it now. This should not happen.
      if (!paperDetails) {
        throw new Error(`Paper ${paperId} information does not exist in API.`);
      }

      if (this.nodeMap.has(paperDetails.id)) {
        this.highlightNode(paperDetails.id);
        return;
      }

      // Ensure we have valid dimensions before positioning
      if (this.width <= 0 || this.height <= 0) {
        this.handleResize();
      }

      const newNode = {
        ...paperDetails,
        x: this.width / 2,
        y: this.height / 2,
      };

      this.addNode(newNode);

      // Show this paper in the panel
      const addedNode = this.nodeMap.get(paperDetails.id);
      if (addedNode) {
        this.showInfoPanel(addedNode);
      }

      // Load paper pools and show initial 2 of each type
      await Promise.all([
        this.loadCitedPapers(paperDetails.id),
        this.loadSemanticPapers(paperDetails.id),
      ]);

      // Show initial papers of each type
      await Promise.all([
        this.expandCitations(paperDetails.id, PaperNetwork.INITIAL_PAPERS_PER_TYPE),
        this.expandSemantic(paperDetails.id, PaperNetwork.INITIAL_PAPERS_PER_TYPE),
      ]);
    } catch (error) {
      console.error("Error adding paper:", error);
      void showDialog(
        "Error adding paper to graph. Please try again.",
        "Add Paper Error",
      );
    } finally {
      this.showLoading(false);
    }
  }

  private updateCurrentNodeCounts(nodeId: string): void {
    if (!this.currentSelectedNode || this.currentSelectedNode.id !== nodeId) {
      return;
    }
    const citedCountEl = requireElement("citedCount");
    const semanticCountEl = requireElement("semanticCount");

    const citedShown = this.citedShown.get(nodeId) ?? 0;
    const citedTotal = this.citedPools.get(nodeId)?.length ?? 0;
    const semanticShown = this.semanticShown.get(nodeId) ?? 0;
    const semanticTotal = this.semanticPools.get(nodeId)?.length ?? 0;

    // Only show counts if papers have been loaded
    citedCountEl.textContent =
      citedTotal > 0 ? `${citedShown}/${citedTotal} shown` : "";
    semanticCountEl.textContent =
      semanticTotal > 0 ? `${semanticShown}/${semanticTotal} shown` : "";

    // Update button states
    this.updateExpandButtonStates(nodeId);
  }

  private updateExpandButtonStates(nodeId: string): void {
    const citedShown = this.citedShown.get(nodeId) ?? 0;
    const citedTotal = this.citedPools.get(nodeId)?.length ?? 0;
    const semanticShown = this.semanticShown.get(nodeId) ?? 0;
    const semanticTotal = this.semanticPools.get(nodeId)?.length ?? 0;

    // Update citation buttons
    this.expandCitedButtons.forEach((button, index) => {
      const additionalCount = index + 1; // +1, +2, +3
      // If papers haven't been loaded yet (total = 0), enable buttons
      // Otherwise, disable if adding would exceed total
      const shouldDisable = citedTotal > 0 && citedShown + additionalCount > citedTotal;
      button.disabled = shouldDisable;

      // Add/remove visual disabled state
      if (shouldDisable) {
        button.classList.add("opacity-50", "cursor-not-allowed");
        button.classList.remove("hover:bg-gray-700");
      } else {
        button.classList.remove("opacity-50", "cursor-not-allowed");
        button.classList.add("hover:bg-gray-700");
      }
    });

    // Update semantic buttons
    this.expandSemanticButtons.forEach((button, index) => {
      const additionalCount = index + 1; // +1, +2, +3
      // If papers haven't been loaded yet (total = 0), enable buttons
      // Otherwise, disable if adding would exceed total
      const shouldDisable =
        semanticTotal > 0 && semanticShown + additionalCount > semanticTotal;
      button.disabled = shouldDisable;

      // Add/remove visual disabled state
      if (shouldDisable) {
        button.classList.add("opacity-50", "cursor-not-allowed");
        button.classList.remove("hover:bg-gray-700");
      } else {
        button.classList.remove("opacity-50", "cursor-not-allowed");
        button.classList.add("hover:bg-gray-700");
      }
    });
  }

  private addNode(paper: D3Node): void {
    if (this.nodeMap.has(paper.id)) return;

    this.nodes.push(paper);
    this.nodeMap.set(paper.id, paper);
    this.updateVisualization();
  }

  private getPaperTypeConfig(type: LinkType): PaperTypeConfig {
    if (type === "cited") {
      return {
        pool: this.citedPools,
        shown: this.citedShown,
        apiCall: this.dataService.getCitedPapers.bind(this.dataService),
        linkType: "cited",
      };
    } else {
      return {
        pool: this.semanticPools,
        shown: this.semanticShown,
        apiCall: this.dataService.getSemanticPapers.bind(this.dataService),
        linkType: "similar",
      };
    }
  }

  private async loadPapers(nodeId: string, type: LinkType): Promise<void> {
    const config = this.getPaperTypeConfig(type);

    if (config.pool.has(nodeId)) return; // Already loaded

    this.showLoading(true, `Loading ${config.linkType}`);

    try {
      const response = await config.apiCall(nodeId, 20);
      config.pool.set(nodeId, response.neighbours);

      console.log(`Loaded ${config.linkType} pool:`, {
        nodeId,
        count: response.neighbours.length,
        papers: response.neighbours.map(p => `${p.id}: ${p.title}`),
      });

      // Update button states if this is the currently selected node
      if (this.currentSelectedNode && this.currentSelectedNode.id === nodeId) {
        this.updateExpandButtonStates(nodeId);
      }
    } catch (error) {
      console.error(`Error loading ${config.linkType}:`, error);
    } finally {
      this.showLoading(false);
    }
  }

  private async expandPapers(
    nodeId: string,
    additionalCount: number,
    type: LinkType,
  ): Promise<void> {
    if (additionalCount <= 0) return;

    const config = this.getPaperTypeConfig(type);

    // Load pool if not already loaded
    await this.loadPapers(nodeId, type);

    const pool = config.pool.get(nodeId) ?? [];
    const currentShown = config.shown.get(nodeId) ?? 0;

    // Early return if all papers are already shown
    if (currentShown >= pool.length) return;

    const newShown = Math.min(currentShown + additionalCount, pool.length);
    const papersToShow = pool.slice(currentShown, newShown);

    console.log(`expand${type}:`, {
      nodeId,
      additionalCount,
      currentShown,
      newShown,
      poolSize: pool.length,
      showingCount: papersToShow.length,
    });

    // Add nodes and links for papers to show
    papersToShow.forEach(paper => {
      // Add node if it doesn't exist
      if (!this.nodeMap.has(paper.id)) {
        // Create D3Node from PaperNeighbour
        const newNode: D3Node = { ...paper };

        // Position new nodes near their parent node
        const parentNode = this.nodeMap.get(nodeId);
        if (parentNode && parentNode.x !== undefined && parentNode.y !== undefined) {
          // Add some randomness to avoid overlapping
          const angle = Math.random() * 2 * Math.PI;
          const distance = 100 + Math.random() * 50;
          newNode.x = parentNode.x + Math.cos(angle) * distance;
          newNode.y = parentNode.y + Math.sin(angle) * distance;
        }
        this.addNode(newNode);
      }

      // Add link if it doesn't exist
      if (!this.linkExists(nodeId, paper.id)) {
        this.addLink({
          source: nodeId,
          target: paper.id,
          type: config.linkType,
          similarity: paper.similarity,
        });
      }
    });

    // Update shown count
    config.shown.set(nodeId, newShown);
    this.expandedNodes.add(nodeId);

    this.updateVisualization();
    this.updateCurrentNodeCounts(nodeId);
  }

  private async loadCitedPapers(nodeId: string): Promise<void> {
    return this.loadPapers(nodeId, "cited");
  }

  private async loadSemanticPapers(nodeId: string): Promise<void> {
    return this.loadPapers(nodeId, "similar");
  }

  private async expandCitations(
    nodeId: string,
    additionalCount: number,
  ): Promise<void> {
    return this.expandPapers(nodeId, additionalCount, "cited");
  }

  private async expandSemantic(nodeId: string, additionalCount: number): Promise<void> {
    return this.expandPapers(nodeId, additionalCount, "similar");
  }

  private addLink(link: GraphLink): void {
    // Check if link already exists
    const exists = this.links.some(
      l => getNodeId(l.source) === link.source && getNodeId(l.target) === link.target,
    );

    if (!exists) {
      this.links.push(link as D3Link);
    }
  }

  private linkExists(sourceId: string, targetId: string): boolean {
    return this.links.some(
      l => getNodeId(l.source) === sourceId && getNodeId(l.target) === targetId,
    );
  }

  private updateVisualization(): void {
    // Update links
    const linkSelection = this.linkGroup
      .selectAll<SVGLineElement, D3Link>(".link")
      .data(this.links, d => `${getNodeId(d.source)}-${getNodeId(d.target)}`);

    linkSelection.exit().remove();

    const linkEnter = linkSelection
      .enter()
      .append("line")
      .attr("class", "link")
      .attr("stroke", d => (d.type === "cited" ? "#ff6b6b" : "#4ecdc4"))
      .attr("stroke-width", d => {
        // Minimum thickness for similarities below 40%
        const minThickness = 4;
        const maxThickness = 12;

        if (d.similarity < 0.4) {
          return minThickness;
        }

        // Scale from 40% to 100% similarity
        const scaledSimilarity = (d.similarity - 0.4) / 0.6;
        return minThickness + scaledSimilarity * (maxThickness - minThickness);
      });

    const linkUpdate = linkEnter.merge(linkSelection);

    // Add hover handlers for links
    linkUpdate
      .on("mouseover", (event: MouseEvent, d: D3Link) => this.showLinkTooltip(event, d))
      .on("mousemove", (event: MouseEvent) => this.updateTooltipPosition(event))
      .on("mouseout", () => this.hideTooltip());

    // Update nodes
    const nodeSelection = this.nodeGroup
      .selectAll<SVGGElement, D3Node>(".node")
      .data(this.nodes, d => d.id);

    nodeSelection.exit().remove();

    const nodeEnter = nodeSelection
      .enter()
      .append("g")
      .attr("class", "node")
      .call(this.createDragBehavior());

    const rectWidth = 160;
    nodeEnter
      .append("rect")
      .attr("class", "node-rect")
      .attr("x", -80)
      .attr("y", -30)
      .attr("width", rectWidth)
      .attr("height", 60)
      .attr("rx", 25)
      .attr("ry", 25)
      .attr("fill", "#2a2a2a")
      .attr("stroke", "#4a9eff")
      .attr("stroke-width", 2);

    const nodeUpdate = nodeEnter.merge(nodeSelection);

    // Remove existing text elements to rebuild them
    nodeUpdate.selectAll(".node-text").remove();

    // Add text with wrapping
    nodeUpdate.each(function (d) {
      // Split text into words and wrap
      const words = d.title.split(/\s+/);
      const lineHeight = 14;
      const fontSize = 13;
      const maxWidth = rectWidth - 16; // more padding for better readability
      const maxLines = 3; // Maximum number of lines

      const text = d3
        .select(this)
        .append("text")
        .attr("class", "node-text")
        .attr("fill", "white")
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "middle")
        .attr("font-size", `${fontSize}px`);

      let line: string[] = [];
      let lineNumber = 0;
      let tspan = text.append("tspan").attr("x", 0).attr("dy", 0);

      for (const word of words) {
        line.push(word);
        tspan.text(line.join(" "));

        // Check if line is too long
        if ((tspan.node()?.getComputedTextLength() ?? 0) > maxWidth) {
          line.pop();

          // Check if we've reached the line limit
          if (lineNumber + 1 >= maxLines) {
            // Add ellipsis to current line
            const currentText = line.join(" ");
            tspan.text(currentText + "...");

            // Keep removing words until the line with ellipsis fits
            while (
              (tspan.node()?.getComputedTextLength() ?? 0) > maxWidth &&
              line.length > 1
            ) {
              line.pop();
              tspan.text(line.join(" ") + "...");
            }
            break;
          }

          tspan.text(line.join(" "));
          line = [word];
          lineNumber++;
          tspan = text.append("tspan").attr("x", 0).attr("dy", lineHeight).text(word);
        }
      }

      // Center the text vertically
      const totalLines = Math.min(lineNumber + 1, maxLines);
      const totalHeight = totalLines * lineHeight;
      text.attr("transform", `translate(0, ${-totalHeight / 2 + lineHeight / 2})`);
    });

    // Clear existing event handlers and add new ones
    nodeUpdate
      .on("click", null)
      .on("mouseover", null)
      .on("mousemove", null)
      .on("mouseout", null);

    nodeUpdate
      .on("click", (event: MouseEvent, d: D3Node) => {
        event.stopPropagation();
        this.showInfoPanel(d);
      })
      .on("mouseover", (event: MouseEvent, d: D3Node) => this.showTooltip(event, d))
      .on("mousemove", (event: MouseEvent) => this.updateTooltipPosition(event))
      .on("mouseout", () => this.hideTooltip());

    // Update simulation
    this.simulation.nodes(this.nodes).on("tick", () => {
      linkUpdate
        .attr("x1", d => getNodePos(d.source).x ?? 0)
        .attr("y1", d => getNodePos(d.source).y ?? 0)
        .attr("x2", d => getNodePos(d.target).x ?? 0)
        .attr("y2", d => getNodePos(d.target).y ?? 0);

      nodeUpdate.attr("transform", d => `translate(${d.x ?? 0},${d.y ?? 0})`);
    });

    this.simulation.force<d3.ForceLink<D3Node, D3Link>>("link")?.links(this.links);
    // Use lower alpha for gentler repositioning that preserves existing layout
    this.simulation.alpha(0.3).restart();
  }

  private createDragBehavior(): d3.DragBehavior<
    SVGGElement,
    D3Node,
    D3Node | d3.SubjectPosition
  > {
    type DragEvent = d3.D3DragEvent<SVGGElement, D3Node, D3Node>;
    return d3
      .drag<SVGGElement, D3Node>()
      .on("start", (event: DragEvent, d: D3Node) => {
        if (!event.active) this.simulation.alphaTarget(0.1).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on("drag", (event: DragEvent, d: D3Node) => {
        d.fx = event.x;
        d.fy = event.y;
      })
      .on("end", (event: DragEvent, d: D3Node) => {
        if (!event.active) this.simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      });
  }

  private showTooltip(event: MouseEvent, node: D3Node): void {
    this.tooltip.textContent = node.title;
    this.tooltip.classList.remove("opacity-0");
    this.tooltip.classList.add("opacity-100");
    this.updateTooltipPosition(event);
  }

  private updateTooltipPosition(event: MouseEvent): void {
    const tooltipRect = this.tooltip.getBoundingClientRect();
    const container = requireElement("graphContainer");
    const containerRect = container.getBoundingClientRect();

    let left = event.clientX - containerRect.left + 15;
    let top = event.clientY - containerRect.top - 10;

    // Adjust if tooltip would go off screen
    if (left + tooltipRect.width > containerRect.width) {
      left = event.clientX - containerRect.left - tooltipRect.width - 15;
    }

    if (top < 0) {
      top = event.clientY - containerRect.top + 25;
    }

    this.tooltip.style.left = `${left}px`;
    this.tooltip.style.top = `${top}px`;
  }

  private hideTooltip(): void {
    this.tooltip.classList.remove("opacity-100");
    this.tooltip.classList.add("opacity-0");
  }

  private showLinkTooltip(event: MouseEvent, link: D3Link): void {
    const linkType = link.type === "cited" ? "Citation" : "Semantic Similarity";
    const strength = (link.similarity * 100).toFixed(1);
    this.tooltip.textContent = `${linkType} (${strength}%)`;
    this.tooltip.classList.remove("opacity-0");
    this.tooltip.classList.add("opacity-100");
    this.updateTooltipPosition(event);
  }

  private showInfoPanel(node: D3Node): void {
    this.currentSelectedNode = node;

    const titleEl = requireElement("infoPanelTitle");
    const idEl = requireElement("infoPanelId");
    const authorsEl = requireElement("infoPanelAuthors");
    const yearEl = requireElement("infoPanelYear");
    const abstractEl = requireElement("infoPanelAbstract");
    const citedCountEl = requireElement("citedCount");
    const semanticCountEl = requireElement("semanticCount");

    titleEl.textContent = node.title;
    idEl.textContent = `ID: ${node.id}`;
    authorsEl.textContent =
      node.authors.length > 0 ? node.authors.join(", ") : "Authors not available";
    yearEl.textContent = node.year ? node.year.toString() : "Year not available";
    abstractEl.textContent = node.abstract || "Abstract not available";

    // Update counts display
    const citedShown = this.citedShown.get(node.id) ?? 0;
    const citedTotal = this.citedPools.get(node.id)?.length ?? 0;
    const semanticShown = this.semanticShown.get(node.id) ?? 0;
    const semanticTotal = this.semanticPools.get(node.id)?.length ?? 0;

    // Only show counts if papers have been loaded
    citedCountEl.textContent =
      citedTotal > 0 ? `${citedShown}/${citedTotal} shown` : "";
    semanticCountEl.textContent =
      semanticTotal > 0 ? `${semanticShown}/${semanticTotal} shown` : "";

    // Update button states
    this.updateExpandButtonStates(node.id);

    console.log("Showing info panel for node:", {
      nodeId: node.id,
      citedShown,
      citedTotal,
      semanticShown,
      semanticTotal,
    });
  }

  private highlightNode(nodeId: string): void {
    this.nodeGroup
      .selectAll(".node")
      // @ts-expect-error Improper typing from D3
      .filter(d => d.id === nodeId)
      .select(".node-rect")
      .transition()
      .duration(300)
      .attr("stroke-width", 5)
      .transition()
      .duration(300)
      .attr("stroke-width", 2);
  }

  private showLoading(show: boolean, message = "Loading..."): void {
    const loadingDiv = requireElement("loading");
    const loadingText = requireElement("loadingText");

    loadingDiv.style.display = show ? "block" : "none";

    if (show) {
      loadingText.textContent = message;
    }
  }

  private clearGraph(): void {
    this.nodes = [];
    this.links = [];
    this.nodeMap.clear();
    this.expandedNodes.clear();
    this.citedPools.clear();
    this.semanticPools.clear();
    this.citedShown.clear();
    this.semanticShown.clear();
    this.currentSelectedNode = null;

    // Clear panel content
    const titleEl = requireElement("infoPanelTitle");
    const idEl = requireElement("infoPanelId");
    const authorsEl = requireElement("infoPanelAuthors");
    const yearEl = requireElement("infoPanelYear");
    const abstractEl = requireElement("infoPanelAbstract");
    const citedCountEl = requireElement("citedCount");
    const semanticCountEl = requireElement("semanticCount");

    titleEl.textContent = "";
    idEl.textContent = "";
    authorsEl.textContent = "";
    yearEl.textContent = "";
    abstractEl.textContent = "";
    citedCountEl.textContent = "";
    semanticCountEl.textContent = "";

    this.hideSearchResults();

    this.linkGroup.selectAll(".link").remove();
    this.nodeGroup.selectAll(".node").remove();

    this.simulation.nodes([]);
    this.simulation.force<d3.ForceLink<D3Node, D3Link>>("link")?.links([]);
  }
}
