import { z } from "zod/v4";

import {
  retryWithBackoff,
  waitForDOM,
  showInitError,
  isMobileDevice,
  showMobileMessage,
} from "../util";
import {
  GraphResult,
  PaperSearchResults,
  PaperSearchItem,
  PartialEvaluationResponse,
} from "./model";
import { renderLatex, getArxivUrl, formatConferenceName } from "./helpers";
import {
  JsonPaperDataset,
  ArxivPaperService,
  PaperEvaluator,
  PartialPaperEvaluator,
  PartialEvaluationParams,
} from "./services";
import { addFooter } from "../footer";

// Extended interface for cached papers
interface CachedPaperSearchItem extends Omit<PaperSearchItem, "arxiv_id"> {
  arxiv_id?: string; // Make optional for cached papers
  _isCached?: boolean;
  _cachedPaperId?: string;
}

class PaperExplorer {
  private allPapers: GraphResult[] = [];
  private filteredPapers: GraphResult[] = [];
  private lastSelectedArxivItem: PaperSearchItem | null = null;
  // private currentTab: "json" | "arxiv" = "json";

  constructor(
    private jsonDataset: JsonPaperDataset,
    private arxivService: ArxivPaperService | null = null,
    private paperEvaluator: PaperEvaluator | null = null,
    private partialEvaluator: PartialPaperEvaluator | null = null,
  ) {}

  async initialize(): Promise<void> {
    this.setupTabs();
    if (this.arxivService) {
      this.setupArxivSearchBar();
      this.setupClearCacheButton();
      this.setupEvaluationModal();
    }
    if (this.partialEvaluator) {
      this.setupAbstractTab();
    }
    this.setupHelpModal();
    await this.loadPapers();
  }

  private async loadPapers(): Promise<void> {
    try {
      const validatedData = await this.jsonDataset.loadDataset();

      const loading = document.getElementById("json-loading");
      const papersContainer = document.getElementById("json-papers-container");

      if (loading) loading.style.display = "none";
      if (papersContainer) papersContainer.classList.remove("hidden");

      this.allPapers = validatedData;
      this.filteredPapers = [...this.allPapers];

      // Store papers in localStorage for detail page access
      localStorage.setItem("papers-dataset", JSON.stringify(validatedData));

      this.displayPapers(this.filteredPapers);
      this.setupSearchBar();

      const resultCount = document.getElementById("json-result-count");
      if (resultCount) {
        resultCount.textContent = `${this.filteredPapers.length} of ${this.allPapers.length} papers`;
      }
    } catch (error) {
      console.error("Error loading papers:", error);
      const loading = document.getElementById("json-loading");
      const errorEl = document.getElementById("json-error");
      const errorMessage = document.getElementById("json-error-message");

      if (loading) loading.style.display = "none";
      if (errorEl) errorEl.classList.remove("hidden");
      if (errorMessage) {
        if (error instanceof z.ZodError) {
          errorMessage.textContent = `Data validation error: ${error.message}`;
        } else {
          errorMessage.textContent =
            error instanceof Error ? error.message : "Unknown error";
        }
      }
    }
  }

  private displayPapers(papers: GraphResult[]): void {
    const container = document.getElementById("json-papers-container");
    if (!container) return;
    container.innerHTML = "";

    if (papers.length === 0) {
      container.innerHTML = `
        <div class="col-span-full text-center text-gray-600 dark:text-gray-400 py-8">
          No papers found matching your search.
        </div>
      `;
      return;
    }

    papers.forEach(graphResult => {
      const paper = graphResult.paper;
      const paperLink = document.createElement("a");
      const encodedId = encodeURIComponent(paper.id);
      paperLink.href = `/paper-hypergraph/pages/detail.html?id=${encodedId}`;
      paperLink.className =
        "block rounded-lg bg-gray-100/50 dark:bg-gray-900/50 border border-gray-300" +
        " dark:border-gray-700 p-6 transition-all duration-200 hover:border-teal-500/50" +
        " hover:-translate-y-1 no-underline";

      // Extract keywords from graph entities
      const keywords = graphResult.graph.entities
        .filter(e => e.type === "keyword")
        .map(e => e.label)
        .slice(0, 5); // Limit to 5 keywords

      paperLink.innerHTML = `
        <div class="mb-3">
          <h3 class="text-lg font-semibold text-black dark:text-white leading-tight">
            ${renderLatex(paper.title)}
          </h3>
          <p class="text-sm text-gray-600 dark:text-gray-400 mt-1">
            ${paper.authors.join(", ")}
          </p>
        </div>

        <p class="text-sm text-gray-700 dark:text-gray-300 mb-4 line-clamp-3">
          ${renderLatex(paper.abstract)}
        </p>

        ${
          keywords.length > 0
            ? `
          <div class="flex flex-wrap gap-2 mb-4">
            ${keywords
              .map(
                keyword => `
                <span class="px-2 py-1 bg-teal-100/70 dark:bg-teal-900/30 text-teal-800
                             dark:text-teal-300 text-xs rounded-md border border-teal-300/50
                             dark:border-teal-700/50">
                  ${keyword}
                </span>
              `,
              )
              .join("")}
          </div>
        `
            : ""
        }

        <div class="flex justify-between items-center text-xs text-gray-600
                    dark:text-gray-400">
          <div class="flex items-center gap-2">
            <span>${formatConferenceName(paper.conference)} (${paper.year})</span>
            ${
              paper.arxiv_id
                ? `
                  <a href="${getArxivUrl(paper.arxiv_id)}" target="_blank"
                     rel="noopener noreferrer"
                     class="text-blue-600 dark:text-blue-400 hover:text-blue-500
                            dark:hover:text-blue-300 underline transition-colors
                            duration-200"
                     onclick="event.stopPropagation();">
                    arXiv
                  </a>
                `
                : ""
            }
          </div>
        </div>
      `;

      container.appendChild(paperLink);
    });
  }

  private substringSearch(query: string, papers: GraphResult[]): GraphResult[] {
    if (!query.trim()) return papers;

    const lowerQuery = query.toLowerCase();

    // Filter and score papers
    const scoredPapers = papers
      .map(paper => {
        const titleMatches = this.countSubstringMatches(
          paper.paper.title.toLowerCase(),
          lowerQuery,
        );
        const authorMatches = paper.paper.authors
          .map(author => this.countSubstringMatches(author.toLowerCase(), lowerQuery))
          .reduce((sum, count) => sum + count, 0);
        const abstractMatches = this.countSubstringMatches(
          paper.paper.abstract.toLowerCase(),
          lowerQuery,
        );

        const score = titleMatches * 10 + authorMatches * 5 + abstractMatches * 1;

        return { paper, score };
      })
      .filter(({ score }) => score > 0)
      .sort((a, b) => b.score - a.score);

    return scoredPapers.map(({ paper }) => paper);
  }

  private countSubstringMatches(text: string, query: string): number {
    if (!text || !query) return 0;

    let count = 0;
    let position = 0;

    while ((position = text.indexOf(query, position)) !== -1) {
      count++;
      position += query.length;
    }

    return count;
  }

  private setupSearchBar(): void {
    const searchInput = document.getElementById("json-search-input");
    const clearButton = document.getElementById("json-clear-search");
    const resultCount = document.getElementById("json-result-count");

    if (!searchInput || !clearButton || !resultCount) return;

    const inputElement = searchInput as HTMLInputElement;
    const clearElement = clearButton as HTMLButtonElement;
    const countElement = resultCount;

    const updateSearch = () => {
      const query = inputElement.value;
      this.filteredPapers = this.substringSearch(query, this.allPapers);
      this.displayPapers(this.filteredPapers);

      countElement.textContent = `${this.filteredPapers.length} of ${this.allPapers.length} papers`;
      clearElement.style.display = query ? "block" : "none";
    };

    inputElement.addEventListener("input", updateSearch);
    clearElement.addEventListener("click", () => {
      inputElement.value = "";
      updateSearch();
      inputElement.focus();
    });

    updateSearch();
  }

  private setupTabs(): void {
    const jsonTab = document.getElementById("json-tab");
    const arxivTab = document.getElementById("arxiv-tab");
    const abstractTab = document.getElementById("abstract-tab");
    const jsonContent = document.getElementById("json-content");
    const arxivContent = document.getElementById("arxiv-content");
    const abstractContent = document.getElementById("abstract-content");

    if (
      !jsonTab ||
      !arxivTab ||
      !abstractTab ||
      !jsonContent ||
      !arxivContent ||
      !abstractContent
    )
      return;

    jsonTab.addEventListener("click", () => this.switchTab("json"));
    arxivTab.addEventListener("click", () => this.switchTab("arxiv"));
    abstractTab.addEventListener("click", () => this.switchTab("abstract"));
  }

  private switchTab(tab: "json" | "arxiv" | "abstract"): void {
    // this.currentTab = tab;

    const jsonTab = document.getElementById("json-tab");
    const arxivTab = document.getElementById("arxiv-tab");
    const abstractTab = document.getElementById("abstract-tab");
    const jsonContent = document.getElementById("json-content");
    const arxivContent = document.getElementById("arxiv-content");
    const abstractContent = document.getElementById("abstract-content");

    if (
      !jsonTab ||
      !arxivTab ||
      !abstractTab ||
      !jsonContent ||
      !arxivContent ||
      !abstractContent
    )
      return;

    // Define active and inactive classes
    const activeClasses =
      "tab-button active cursor-pointer rounded-md bg-white px-4 py-2 text-sm font-semibold" +
      " text-teal-600 shadow-sm transition-all duration-200 dark:bg-gray-700 dark:text-teal-400";
    const inactiveClasses =
      "tab-button cursor-pointer rounded-md px-4 py-2 text-sm font-semibold text-gray-600" +
      " transition-all duration-200 hover:text-gray-800 dark:text-gray-400 dark:hover:text-gray-200";

    // Reset all tabs to inactive
    jsonTab.className = inactiveClasses;
    arxivTab.className = inactiveClasses;
    abstractTab.className = inactiveClasses;

    // Hide all content
    jsonContent.classList.add("hidden");
    arxivContent.classList.add("hidden");
    abstractContent.classList.add("hidden");

    // Activate the selected tab and show its content
    if (tab === "json") {
      jsonTab.className = activeClasses;
      jsonContent.classList.remove("hidden");
    } else if (tab === "arxiv") {
      arxivTab.className = activeClasses;
      arxivContent.classList.remove("hidden");
    } else {
      // tab === "abstract"
      abstractTab.className = activeClasses;
      abstractContent.classList.remove("hidden");
      // Load previous evaluations when switching to abstract tab
      this.loadPreviousAbstractEvaluations();
    }
  }

  private async searchArxivPapers(query: string): Promise<void> {
    if (!query.trim()) return;

    const loading = document.getElementById("arxiv-loading");
    const errorEl = document.getElementById("arxiv-error");
    const errorMessage = document.getElementById("arxiv-error-message");
    const resultCount = document.getElementById("arxiv-result-count");
    const papersContainer = document.getElementById("arxiv-papers-container");

    try {
      if (errorEl) errorEl.classList.add("hidden");
      if (loading) loading.classList.remove("hidden");
      if (papersContainer) papersContainer.classList.add("hidden");

      if (!this.arxivService) {
        throw new Error("ArXiv service not available");
      }
      const results = await this.arxivService.searchPapers(query);

      if (loading) loading.classList.add("hidden");
      if (papersContainer) papersContainer.classList.remove("hidden");

      this.displayArxivPapers(results);

      if (resultCount) {
        resultCount.textContent = `Found ${results.total} papers for "${results.query}"`;
      }
    } catch (error) {
      console.error("Error searching arXiv:", error);
      if (loading) loading.classList.add("hidden");
      if (errorEl) errorEl.classList.remove("hidden");
      if (errorMessage) {
        errorMessage.textContent =
          error instanceof Error ? error.message : "Unknown error";
      }
    }
  }

  private displayArxivPapers(searchResults: PaperSearchResults): void {
    const container = document.getElementById("arxiv-papers-container");
    if (!container) return;
    container.innerHTML = "";

    if (searchResults.items.length === 0) {
      container.innerHTML = `
        <div class="col-span-full text-center text-gray-600 dark:text-gray-400 py-8">
          No papers found for "${searchResults.query}".
        </div>
      `;
      return;
    }

    searchResults.items.forEach(item => {
      const paperDiv = document.createElement("div");
      paperDiv.className =
        "block rounded-lg bg-gray-100/50 dark:bg-gray-900/50 border border-gray-300" +
        " dark:border-gray-700 p-6 transition-all duration-200 hover:border-teal-500/50" +
        " hover:-translate-y-1 cursor-pointer";

      paperDiv.innerHTML = `
        <div class="mb-3">
          <h3 class="text-lg font-semibold text-black dark:text-white leading-tight">
            ${renderLatex(item.title)}
          </h3>
          <p class="text-sm text-gray-600 dark:text-gray-400 mt-1">
            ${item.authors.join(", ")}
          </p>
        </div>

        <p class="text-sm text-gray-700 dark:text-gray-300 mb-4 line-clamp-3">
          ${renderLatex(item.abstract)}
        </p>

        <div class="flex justify-between items-center text-xs text-gray-600
                    dark:text-gray-400">
          <div class="flex items-center gap-2">
            <span>${item.year ?? "N/A"}</span>
            <a href="${getArxivUrl(item.arxiv_id)}" target="_blank"
               rel="noopener noreferrer"
               class="text-blue-600 dark:text-blue-400 hover:text-blue-500
                      dark:hover:text-blue-300 underline transition-colors
                      duration-200"
               onclick="event.stopPropagation();">
              View on arXiv
            </a>
          </div>
        </div>
      `;

      paperDiv.addEventListener("click", () => this.handleArxivPaperSelection(item));

      container.appendChild(paperDiv);
    });
  }

  private setupArxivSearchBar(): void {
    const searchForm = document.getElementById("arxiv-search-form");
    const searchInput = document.getElementById("arxiv-search-input");
    const clearButton = document.getElementById("arxiv-clear-search");

    if (!searchForm || !searchInput || !clearButton) return;

    const formElement = searchForm as HTMLFormElement;
    const inputElement = searchInput as HTMLInputElement;
    const clearElement = clearButton as HTMLButtonElement;

    const performSearch = () => {
      const query = inputElement.value.trim();
      if (query) {
        void this.searchArxivPapers(query);
      }
    };

    const updateClearButton = () => {
      const query = inputElement.value;
      clearElement.style.display = query ? "block" : "none";
    };

    // Handle form submission
    formElement.addEventListener("submit", event => {
      event.preventDefault();
      performSearch();
    });

    // Handle input changes (only for clear button visibility)
    inputElement.addEventListener("input", updateClearButton);

    // Handle clear button click
    clearElement.addEventListener("click", () => {
      inputElement.value = "";
      clearElement.style.display = "none";
      inputElement.focus();

      // Show cached papers when search is cleared
      this.displayCachedPapersIfEmpty();
    });

    // Initial update
    updateClearButton();

    // Show cached papers initially when search is empty
    this.displayCachedPapersIfEmpty();
  }

  /**
   * Get all cached papers from localStorage
   */
  private getCachedPapers(): GraphResult[] {
    const cacheKeys = Object.keys(localStorage).filter(key =>
      key.startsWith("paper-cache-"),
    );

    const cachedPapers: GraphResult[] = [];

    for (const key of cacheKeys) {
      try {
        const cachedData = localStorage.getItem(key);
        if (cachedData) {
          const graphResult = JSON.parse(cachedData) as GraphResult;
          cachedPapers.push(graphResult);
        }
      } catch (error) {
        console.warn(`Failed to parse cached paper ${key}:`, error);
      }
    }

    // Sort by title for consistent display
    return cachedPapers.sort((a, b) => a.paper.title.localeCompare(b.paper.title));
  }

  /**
   * Display cached papers as cards if search is empty
   */
  private displayCachedPapersIfEmpty(): void {
    const searchInput = document.getElementById("arxiv-search-input");
    if (!searchInput || (searchInput as HTMLInputElement).value.trim()) return;

    const cachedPapers = this.getCachedPapers();
    const papersContainer = document.getElementById("arxiv-papers-container");
    const resultCount = document.getElementById("arxiv-result-count");

    if (cachedPapers.length > 0) {
      // Make container visible
      if (papersContainer) papersContainer.classList.remove("hidden");

      // Convert cached papers to PaperSearchResults format to reuse existing display logic
      const searchResults = this.convertCachedPapersToSearchResults(cachedPapers);
      this.displayArxivPapers(searchResults as PaperSearchResults);

      // Update result count for cached papers
      if (resultCount) {
        resultCount.textContent =
          `${cachedPapers.length}` +
          ` previous evaluation${cachedPapers.length === 1 ? "" : "s"}`;
      }
    } else {
      // No cached papers - clear the display
      if (papersContainer) {
        papersContainer.innerHTML = "";
        papersContainer.classList.add("hidden");
      }
      if (resultCount) {
        resultCount.textContent = "Enter a search query and click Search";
      }
    }
  }

  /**
   * Convert cached GraphResult objects to PaperSearchResults format
   */
  private convertCachedPapersToSearchResults(
    cachedPapers: GraphResult[],
  ): Omit<PaperSearchResults, "items"> & { items: CachedPaperSearchItem[] } {
    const items: CachedPaperSearchItem[] = cachedPapers.map(graphResult => {
      const paper = graphResult.paper;
      return {
        title: paper.title,
        abstract: paper.abstract,
        authors: paper.authors,
        year: paper.year,
        ...(paper.arxiv_id && { arxiv_id: paper.arxiv_id }),
        // Add a flag to identify cached papers for different click behavior
        _isCached: true,
        _cachedPaperId: paper.id,
      };
    });

    return {
      query: "cached",
      total: items.length,
      items: items,
    };
  }

  private handleArxivPaperSelection(item: PaperSearchItem): void {
    // Handle cached papers differently
    const cachedItem = item as CachedPaperSearchItem;
    if (cachedItem._isCached && cachedItem._cachedPaperId) {
      const encodedId = encodeURIComponent(cachedItem._cachedPaperId);
      window.location.href = `/paper-hypergraph/pages/detail.html?id=${encodedId}`;
      return;
    }

    if (!this.arxivService) {
      console.error("ArXiv service not available");
      return;
    }

    // Store the selected item for potential retry
    this.lastSelectedArxivItem = item;

    // Check if paper is already cached first
    console.log(`[Cache] Checking cache for arXiv paper: ${item.arxiv_id}`);

    // Check if paper is already cached
    // Only check keys that match the paper cache pattern (paper-cache-{hash})
    const cacheKeys = Object.keys(localStorage).filter(key =>
      key.startsWith("paper-cache-"),
    );
    console.log(`[Cache] Found ${cacheKeys.length} cached papers in localStorage`);

    for (const key of cacheKeys) {
      try {
        const cachedData = localStorage.getItem(key);
        if (cachedData) {
          const graphResult = JSON.parse(cachedData) as GraphResult;
          console.log(
            `[Cache] Checking ${key}: arxiv_id=${graphResult.paper.arxiv_id}`,
          );

          if (graphResult.paper.arxiv_id === item.arxiv_id) {
            // Paper found in cache, navigate directly
            console.log(
              `[Cache] HIT! Found paper ${item.arxiv_id} in cache with key ${key}`,
            );
            const encodedId = encodeURIComponent(graphResult.paper.id);
            window.location.href = `/paper-hypergraph/pages/detail.html?id=${encodedId}`;
            return;
          }
        }
      } catch (e) {
        // Invalid cache entry, continue checking
        console.warn(`[Cache] Invalid cache entry ${key}:`, e);
      }
    }

    // Paper not in cache, show evaluation settings modal
    console.log(`[Cache] MISS! Paper ${item.arxiv_id} not found in cache`);

    this.showEvaluationSettingsModal(item);
  }

  private showEvaluationModal(title: string): void {
    // Remove any existing modal
    this.hideEvaluationModal();

    const modal = document.createElement("div");
    modal.id = "evaluation-modal";
    modal.className =
      "fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm";

    modal.innerHTML = `
      <div class="mx-4 w-96 rounded-lg bg-white p-6 shadow-xl dark:bg-gray-800">
        <div class="text-center">
          <h3 class="mb-2 text-lg font-semibold text-gray-900 dark:text-white">
            Evaluating Paper
          </h3>
          <p class="mb-4 text-sm text-gray-600 dark:text-gray-400 line-clamp-2">
            ${renderLatex(title)}
          </p>

          <!-- Progress Bar -->
          <div class="mb-4 w-full bg-gray-200 rounded-full h-2 dark:bg-gray-700">
            <div id="progress-bar" class="bg-teal-600 h-2 rounded-full transition-all duration-300"
                 style="width: 0%">
            </div>
          </div>

          <!-- Progress Text -->
          <div class="mb-2 h-12 flex items-center justify-center px-2">
            <p id="progress-text" class="text-sm text-gray-700 dark:text-gray-300 text-center leading-tight">
              Starting evaluation...
            </p>
          </div>

          <!-- Progress Percentage -->
          <p id="progress-percentage" class="text-xs text-gray-500 dark:text-gray-500">
            0%
          </p>

          <!-- Cancel Button -->
          <button id="cancel-arxiv-evaluation"
                  class="mt-4 px-4 py-2 text-sm bg-gray-500 hover:bg-gray-600 text-white
                         rounded-lg transition-colors">
            Cancel
          </button>
        </div>
      </div>
    `;

    document.body.appendChild(modal);

    // Add cancel button handler
    const cancelButton = document.getElementById("cancel-arxiv-evaluation");
    if (cancelButton) {
      cancelButton.addEventListener("click", () => {
        if (this.paperEvaluator) {
          this.paperEvaluator.stopEvaluation();
        }
        this.hideEvaluationModal();
      });
    }
  }

  private updateEvaluationProgress(message: string, progress: number): void {
    const progressBar = document.getElementById("progress-bar");
    const progressText = document.getElementById("progress-text");
    const progressPercentage = document.getElementById("progress-percentage");

    if (progressBar) {
      progressBar.style.width = `${progress}%`;
    }
    if (progressText) {
      progressText.textContent = message;
    }
    if (progressPercentage) {
      progressPercentage.textContent = `${Math.round(progress)}%`;
    }
  }

  private showEvaluationError(errorMessage: string): void {
    const modal = document.getElementById("evaluation-modal");
    if (!modal) return;

    const modalContent = modal.querySelector("div");
    if (!modalContent) return;

    modalContent.innerHTML = `
      <div class="text-center">
        <div class="mb-4">
          <div class="mx-auto h-12 w-12 rounded-full bg-red-100 dark:bg-red-900 flex
                      items-center justify-center">
            <svg class="h-6 w-6 text-red-600 dark:text-red-400" fill="none"
                 viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                    d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.962-.833-2.732 0L4.082 18.5c-.77.833.192 2.5 1.732 2.5z"
              />
            </svg>
          </div>
        </div>
        <h3 class="mb-2 text-lg font-semibold text-gray-900 dark:text-white">
          Evaluation Failed
        </h3>
        <p class="mb-6 text-sm text-gray-600 dark:text-gray-400">
          ${errorMessage}
        </p>

        <div class="flex gap-3 justify-center">
          <button id="retry-evaluation"
                  class="px-4 py-2 text-sm bg-teal-600 hover:bg-teal-700 text-white
                         rounded-lg transition-colors">
            Try Again
          </button>
          <button id="dismiss-error"
                  class="px-4 py-2 text-sm bg-gray-500 hover:bg-gray-600 text-white
                         rounded-lg transition-colors">
            Dismiss
          </button>
        </div>
      </div>
    `;

    // Add event listeners for the buttons
    const retryButton = document.getElementById("retry-evaluation");
    const dismissButton = document.getElementById("dismiss-error");

    if (retryButton) {
      retryButton.addEventListener("click", () => {
        this.hideEvaluationModal();
        // Restart the evaluation with the same item
        if (this.lastSelectedArxivItem) {
          this.handleArxivPaperSelection(this.lastSelectedArxivItem);
        }
      });
    }

    if (dismissButton) {
      dismissButton.addEventListener("click", () => this.hideEvaluationModal());
    }
  }

  private hideEvaluationModal(): void {
    const modal = document.getElementById("evaluation-modal");
    if (modal) {
      modal.remove();
    }
  }

  private setupClearCacheButton(): void {
    const clearButton = document.getElementById("clear-cache");
    if (!clearButton) return;

    clearButton.addEventListener("click", () => {
      // Find all cache keys
      const cacheKeys = Object.keys(localStorage).filter(key =>
        key.startsWith("paper-cache-"),
      );

      if (cacheKeys.length === 0) {
        alert("No previous results found.");
        return;
      }

      if (
        confirm(`Clear ${cacheKeys.length} previous results? This cannot be undone.`)
      ) {
        // Remove all cache keys
        cacheKeys.forEach(key => localStorage.removeItem(key));

        // Refresh the cached papers display if currently showing
        const searchInput = document.getElementById("arxiv-search-input");
        if (searchInput && !(searchInput as HTMLInputElement).value.trim()) {
          this.displayCachedPapersIfEmpty();
        }
      }
    });
  }

  private async requestNotificationPermission(): Promise<void> {
    if (!("Notification" in window)) {
      console.log("This browser does not support notifications");
      return;
    }

    if (Notification.permission === "default") {
      try {
        await Notification.requestPermission();
      } catch (error) {
        console.warn("Failed to request notification permission:", error);
      }
    }
  }

  private showCompletionNotification(paperTitle: string): void {
    if (!("Notification" in window) || Notification.permission !== "granted") {
      return;
    }

    try {
      const titleTrunc = `${paperTitle.substring(0, 100)}${paperTitle.length > 100 ? "..." : ""}`;
      const notification = new Notification("Paper Evaluation Complete", {
        body: `Finished evaluating: ${titleTrunc}`,
        icon: "/favicon.ico",
        tag: "paper-evaluation-complete",
        requireInteraction: false,
        silent: false,
      });

      // Auto-close notification after 5 seconds
      setTimeout(() => notification.close(), 5000);

      // Handle notification click
      notification.onclick = () => {
        window.focus();
        notification.close();
      };
    } catch (error) {
      console.warn("Failed to show notification:", error);
    }
  }

  private setupEvaluationModal(): void {
    const modal = document.getElementById("evaluation-settings-modal");
    const form = document.getElementById("evaluation-settings-form");
    const cancelButton = document.getElementById("cancel-evaluation");
    const paperTitleEl = document.getElementById("evaluation-paper-title");

    if (!modal || !form || !cancelButton || !paperTitleEl) {
      console.warn("Evaluation modal elements not found");
      return;
    }

    // Handle cancel button
    cancelButton.addEventListener("click", () => this.hideEvaluationSettingsModal());

    // Handle clicking outside modal
    modal.addEventListener("click", e => {
      if (e.target === modal) {
        this.hideEvaluationSettingsModal();
      }
    });

    // Handle form submission
    form.addEventListener("submit", e => {
      e.preventDefault();
      void this.handleEvaluationSubmit();
    });

    // Handle Escape key
    document.addEventListener("keydown", e => {
      if (e.key === "Escape" && !modal.classList.contains("hidden")) {
        this.hideEvaluationSettingsModal();
      }
    });
  }

  private setupHelpModal(): void {
    const helpButton = document.getElementById("help-button");
    const helpModal = document.getElementById("help-modal");
    const helpModalClose = document.getElementById("help-modal-close");

    if (!helpButton || !helpModal || !helpModalClose) {
      console.warn("Help modal elements not found");
      return;
    }

    // Check if user has seen the help modal before
    const hasSeenHelp = localStorage.getItem("paper-explorer-help-seen");
    if (!hasSeenHelp) {
      // Show modal for first-time users
      helpModal.classList.remove("hidden");
      helpModal.classList.add("flex");
    }

    // Handle help button click
    helpButton.addEventListener("click", () => {
      helpModal.classList.remove("hidden");
      helpModal.classList.add("flex");
    });

    // Handle close button click
    helpModalClose.addEventListener("click", () => {
      this.hideHelpModal();
    });

    // Handle clicking outside modal
    helpModal.addEventListener("click", e => {
      if (e.target === helpModal) {
        this.hideHelpModal();
      }
    });

    // Handle Escape key
    document.addEventListener("keydown", e => {
      if (e.key === "Escape" && !helpModal.classList.contains("hidden")) {
        this.hideHelpModal();
      }
    });
  }

  private hideHelpModal(): void {
    const helpModal = document.getElementById("help-modal");
    if (helpModal) {
      helpModal.classList.add("hidden");
      helpModal.classList.remove("flex");
      // Mark as seen
      localStorage.setItem("paper-explorer-help-seen", "true");
    }
  }

  private showEvaluationSettingsModal(item: PaperSearchItem): void {
    const modal = document.getElementById("evaluation-settings-modal");
    const paperTitleEl = document.getElementById("evaluation-paper-title");

    if (!modal || !paperTitleEl) {
      console.error("Evaluation settings modal elements not found");
      return;
    }

    // Store the item for later use
    this.lastSelectedArxivItem = item;

    // Update modal title with paper info
    paperTitleEl.textContent = `Configure evaluation for: ${item.title}`;

    // Show modal
    modal.classList.remove("hidden");
    modal.classList.add("flex");
  }

  private hideEvaluationSettingsModal(): void {
    const modal = document.getElementById("evaluation-settings-modal");
    if (modal) {
      modal.classList.add("hidden");
      modal.classList.remove("flex");
    }
  }

  private async handleEvaluationSubmit(): Promise<void> {
    if (!this.lastSelectedArxivItem || !this.paperEvaluator) {
      console.error("No selected item or evaluator not available");
      return;
    }

    // Get form values
    const kRefsInput = document.getElementById("k_refs");
    const recommendationsInput = document.getElementById("recommendations");
    const relatedInput = document.getElementById("related");
    const llmModelSelect = document.getElementById("llm_model");
    const filterByDateCheckbox = document.getElementById("filter_by_date");

    if (
      !kRefsInput ||
      !recommendationsInput ||
      !relatedInput ||
      !llmModelSelect ||
      !filterByDateCheckbox
    ) {
      console.error("Form elements not found");
      return;
    }

    // Parse and validate form values
    const params = {
      id: this.lastSelectedArxivItem.arxiv_id,
      title: this.lastSelectedArxivItem.title,
      k_refs: parseInt((kRefsInput as HTMLInputElement).value, 10),
      recommendations: parseInt((recommendationsInput as HTMLInputElement).value, 10),
      related: parseInt((relatedInput as HTMLInputElement).value, 10),
      llm_model: (llmModelSelect as HTMLSelectElement).value as
        | "gpt-4o"
        | "gpt-4o-mini"
        | "gemini-2.0-flash",
      filter_by_date: (filterByDateCheckbox as HTMLInputElement).checked,
      seed: 0,
    };

    // Validate parameters
    if (params.k_refs < 10 || params.k_refs > 50) {
      alert("References to analyse must be between 10 and 50");
      return;
    }
    if (params.recommendations < 5 || params.recommendations > 50) {
      alert("Recommended papers must be between 5 and 50");
      return;
    }
    if (params.related < 1 || params.related > 10) {
      alert("Related papers per type must be between 1 and 10");
      return;
    }

    // Hide settings modal and show progress modal
    this.hideEvaluationSettingsModal();

    // Request notification permission
    await this.requestNotificationPermission();

    // Show progress modal
    this.showEvaluationModal(params.title);

    try {
      // Set up progress callbacks
      this.paperEvaluator.onConnected = (message: string) => {
        console.log("Connected:", message);
        this.updateEvaluationProgress(message, 0);
      };

      this.paperEvaluator.onProgress = (message: string, progress: number) => {
        console.log(`Progress: ${message} (${progress.toFixed(1)}%)`);
        this.updateEvaluationProgress(message, progress);
      };

      this.paperEvaluator.onError = (error: string) => {
        console.error("Evaluation error:", error);
        this.showEvaluationError(error);
      };

      this.paperEvaluator.onConnectionError = (event: Event) => {
        console.error("Connection error:", event);
        this.showEvaluationError("Connection lost. Please try again.");
      };

      // Start evaluation with custom parameters
      const evalResult = await this.paperEvaluator.startEvaluation(params);

      // Update progress to 100% on completion
      this.updateEvaluationProgress("Evaluation complete", 100);

      // Store the result in localStorage cache
      const paperId = evalResult.result.paper.id;
      console.log(
        `[Cache] Storing paper ${params.id} in cache with key paper-cache-${paperId}`,
      );
      localStorage.setItem(`paper-cache-${paperId}`, JSON.stringify(evalResult.result));

      // Show completion notification
      this.showCompletionNotification(params.title);

      // Navigate to detail page
      const encodedId = encodeURIComponent(paperId);
      window.location.href = `/paper-hypergraph/pages/detail.html?id=${encodedId}`;
    } catch (error) {
      // Check if error is due to user cancellation
      if (error instanceof Error && error.message.includes("cancelled by user")) {
        console.log("Evaluation cancelled by user");
        // Modal is already hidden by cancel button handler, no need to show error
        return;
      }

      console.error("Error evaluating paper:", error);
      this.showEvaluationError("Failed to evaluate paper. Please try again.");
    }
  }

  // TODO: The following abstract tab methods duplicate significant logic from arXiv tab methods.
  // Consider refactoring to share common functionality in a future iteration.

  /**
   * Setup Abstract tab functionality including form submission and cache management
   */
  private setupAbstractTab(): void {
    this.setupAbstractForm();
    this.setupAbstractClearCache();
    this.loadPreviousAbstractEvaluations();
  }

  /**
   * Setup the abstract evaluation form submission
   */
  private setupAbstractForm(): void {
    const form = document.getElementById(
      "abstract-evaluation-form",
    ) as HTMLFormElement | null;
    if (!form) return;

    form.addEventListener("submit", e => {
      e.preventDefault();

      const titleElement = document.getElementById(
        "abstract-title",
      ) as HTMLTextAreaElement | null;
      const abstractElement = document.getElementById(
        "abstract-text",
      ) as HTMLTextAreaElement | null;
      const recommendationsElement = document.getElementById(
        "abstract-recommendations",
      ) as HTMLInputElement | null;
      const relatedElement = document.getElementById(
        "abstract-related",
      ) as HTMLInputElement | null;
      const llmModelElement = document.getElementById(
        "abstract-llm-model",
      ) as HTMLSelectElement | null;

      if (
        !titleElement ||
        !abstractElement ||
        !recommendationsElement ||
        !relatedElement ||
        !llmModelElement
      ) {
        console.error("Required form elements not found");
        return;
      }

      void this.handleAbstractFormSubmission(
        titleElement,
        abstractElement,
        recommendationsElement,
        relatedElement,
        llmModelElement,
      );
    });
  }

  /**
   * Handle abstract form submission
   */
  private async handleAbstractFormSubmission(
    titleElement: HTMLTextAreaElement,
    abstractElement: HTMLTextAreaElement,
    recommendationsElement: HTMLInputElement,
    relatedElement: HTMLInputElement,
    llmModelElement: HTMLSelectElement,
  ): Promise<void> {
    const title = titleElement.value.trim();
    const abstract = abstractElement.value.trim();

    if (!title || !abstract) {
      this.showAbstractError("Please provide both title and abstract.");
      return;
    }

    const params: PartialEvaluationParams = {
      title,
      abstract,
      recommendations: parseInt(recommendationsElement.value) || 20,
      related: parseInt(relatedElement.value) || 3,
      llm_model: llmModelElement.value || "gpt-4o-mini",
    };

    await this.evaluateAbstract(params);
  }

  /**
   * Setup clear cache button for abstract evaluations
   */
  private setupAbstractClearCache(): void {
    const clearButton = document.getElementById("clear-abstract-cache");
    if (!clearButton) return;

    clearButton.addEventListener("click", () => {
      // Remove all partial evaluation cache entries
      const keys = Object.keys(localStorage);
      const partialKeys = keys.filter(key =>
        key.startsWith("partial-evaluation-cache-"),
      );

      partialKeys.forEach(key => localStorage.removeItem(key));
      localStorage.removeItem("partial-evaluations-list");

      this.loadPreviousAbstractEvaluations();

      // Show brief notification
      this.showAbstractError("Previous evaluations cleared.", false);
      setTimeout(() => this.hideAbstractError(), 2000);
    });
  }

  /**
   * Load and display previous abstract evaluations
   */
  private loadPreviousAbstractEvaluations(): void {
    const container = document.getElementById("abstract-results-container");
    const noResults = document.getElementById("abstract-no-results");

    if (!container || !noResults) return;

    // Get list of evaluation IDs from localStorage
    const evaluationsList = localStorage.getItem("partial-evaluations-list");
    const evaluationIds: string[] = evaluationsList
      ? (JSON.parse(evaluationsList) as string[])
      : [];

    container.innerHTML = "";

    if (evaluationIds.length === 0) {
      container.classList.add("hidden");
      noResults.classList.remove("hidden");
      return;
    }

    noResults.classList.add("hidden");
    container.classList.remove("hidden");

    // Load and display each evaluation
    const evaluations: PartialEvaluationResponse[] = [];

    for (const id of evaluationIds) {
      try {
        const cached = localStorage.getItem(`partial-evaluation-cache-${id}`);
        if (cached) {
          const evaluation = JSON.parse(cached) as PartialEvaluationResponse;
          evaluations.push(evaluation);
        }
      } catch (error) {
        console.warn(`Failed to load cached evaluation ${id}:`, error);
      }
    }

    // Sort by most recent (assuming IDs are chronological)
    evaluations.sort((a, b) => b.id.localeCompare(a.id));

    evaluations.forEach(evaluation => {
      const card = this.createAbstractEvaluationCard(evaluation);
      container.appendChild(card);
    });
  }

  /**
   * Create a card element for displaying an abstract evaluation result
   */
  private createAbstractEvaluationCard(
    evaluation: PartialEvaluationResponse,
  ): HTMLElement {
    const card = document.createElement("div");
    card.className =
      "rounded-lg border border-gray-300 bg-gray-50/50 p-4 transition-all " +
      "duration-200 hover:border-teal-500/50 dark:border-gray-700 dark:bg-gray-800/50";

    const probabilityPercent = evaluation.probability
      ? Math.round(evaluation.probability * 100)
      : null;
    const noveltyText = evaluation.label === 1 ? "Novel" : "Not Novel";
    const noveltyColor =
      evaluation.label === 1
        ? "text-green-600 dark:text-green-400"
        : "text-red-600 dark:text-red-400";

    card.innerHTML = `
      <div class="cursor-pointer" onclick="window.location.href='/paper-hypergraph/pages/partial-detail.html?id=${evaluation.id}'">
        <h4 class="mb-2 line-clamp-2 font-semibold text-gray-900 dark:text-gray-100">
          ${renderLatex(evaluation.title)}
        </h4>
        <p class="mb-3 line-clamp-3 text-sm text-gray-600 dark:text-gray-400">
          ${evaluation.abstract.substring(0, 150)}${evaluation.abstract.length > 150 ? "..." : ""}
        </p>
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-2">
            <span class="text-xs font-medium ${noveltyColor}">
              ${noveltyText}
            </span>
            ${
              probabilityPercent !== null
                ? `
              <span class="text-xs text-gray-500 dark:text-gray-500">
                (${probabilityPercent}%)
              </span>
            `
                : ""
            }
          </div>
          <div class="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-500">
            <span>${evaluation.keywords.length} keywords</span>
            <span>•</span>
            <span>${evaluation.related.length} related</span>
          </div>
        </div>
        <div class="mt-2 flex flex-wrap gap-1">
          ${evaluation.keywords
            .slice(0, 3)
            .map(
              keyword => `
            <span class="inline-block rounded-md bg-teal-100/70 px-2 py-0.5 text-xs text-teal-800
                   dark:bg-teal-900/30 dark:text-teal-300">
              ${keyword}
            </span>
          `,
            )
            .join("")}
          ${
            evaluation.keywords.length > 3
              ? `
            <span class="text-xs text-gray-500 dark:text-gray-500">
              +${evaluation.keywords.length - 3} more
            </span>
          `
              : ""
          }
        </div>
      </div>
    `;

    return card;
  }

  /**
   * Evaluate an abstract using the partial evaluation service
   */
  private async evaluateAbstract(params: PartialEvaluationParams): Promise<void> {
    if (!this.partialEvaluator) {
      this.showAbstractError("Abstract evaluation service not available.");
      return;
    }

    try {
      // Show progress modal
      this.showAbstractEvaluationModal(params.title);

      // Setup progress callbacks
      this.partialEvaluator.onConnected = message => {
        this.updateAbstractEvaluationProgress(message, 0);
      };

      this.partialEvaluator.onProgress = (message, progress) => {
        this.updateAbstractEvaluationProgress(message, progress);
      };

      this.partialEvaluator.onError = error => {
        this.hideAbstractEvaluationModal();
        this.showAbstractError(`Evaluation failed: ${error}`);
      };

      this.partialEvaluator.onConnectionError = () => {
        this.hideAbstractEvaluationModal();
        this.showAbstractError("Connection lost during evaluation. Please try again.");
      };

      // Start evaluation
      const result = await this.partialEvaluator.startEvaluation(params);

      // Update progress to 100% on completion
      this.updateAbstractEvaluationProgress("Evaluation complete", 100);

      // Hide progress modal
      this.hideAbstractEvaluationModal();

      // Cache the result
      localStorage.setItem(
        `partial-evaluation-cache-${result.id}`,
        JSON.stringify(result),
      );

      // Update the evaluations list
      const currentList = localStorage.getItem("partial-evaluations-list");
      const evaluationIds: string[] = currentList
        ? (JSON.parse(currentList) as string[])
        : [];
      if (!evaluationIds.includes(result.id)) {
        evaluationIds.push(result.id);
        localStorage.setItem("partial-evaluations-list", JSON.stringify(evaluationIds));
      }

      // Show completion notification
      this.showAbstractCompletionNotification(params.title);

      // Navigate to partial detail page
      window.location.href = `/paper-hypergraph/pages/partial-detail.html?id=${result.id}`;
    } catch (error) {
      // Check if error is due to user cancellation
      if (error instanceof Error && error.message.includes("cancelled by user")) {
        console.log("Abstract evaluation cancelled by user");
        // Modal is already hidden by cancel button handler, no need to show error
        return;
      }

      this.hideAbstractEvaluationModal();
      console.error("Error evaluating abstract:", error);
      this.showAbstractError("Failed to evaluate abstract. Please try again.");
    }
  }

  /**
   * Show progress modal for abstract evaluation
   */
  private showAbstractEvaluationModal(title: string): void {
    // Remove any existing modal
    this.hideAbstractEvaluationModal();

    const modal = document.createElement("div");
    modal.id = "abstract-evaluation-modal";
    modal.className =
      "fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm";

    modal.innerHTML = `
      <div class="mx-4 w-96 rounded-lg bg-white p-6 shadow-xl dark:bg-gray-800">
        <div class="text-center">
          <h3 class="mb-2 text-lg font-semibold text-gray-900 dark:text-white">
            Evaluating Abstract
          </h3>
          <p class="mb-4 text-sm text-gray-600 dark:text-gray-400 line-clamp-2">
            ${renderLatex(title)}
          </p>

          <!-- Progress Bar -->
          <div class="mb-4 w-full bg-gray-200 rounded-full h-2 dark:bg-gray-700">
            <div id="abstract-progress-bar" class="bg-teal-600 h-2 rounded-full transition-all duration-300"
                 style="width: 0%">
            </div>
          </div>

          <!-- Progress Text -->
          <div class="mb-2 h-12 flex items-center justify-center px-2">
            <p id="abstract-progress-text" class="text-sm text-gray-700 dark:text-gray-300 text-center leading-tight">
              Starting evaluation...
            </p>
          </div>

          <!-- Progress Percentage -->
          <p id="abstract-progress-percentage" class="text-xs text-gray-500 dark:text-gray-500">
            0%
          </p>

          <!-- Cancel Button -->
          <button id="cancel-abstract-evaluation"
                  class="mt-4 px-4 py-2 text-sm bg-gray-500 hover:bg-gray-600 text-white
                         rounded-lg transition-colors">
            Cancel
          </button>
        </div>
      </div>
    `;

    document.body.appendChild(modal);

    // Add cancel button handler
    const cancelButton = document.getElementById("cancel-abstract-evaluation");
    if (cancelButton) {
      cancelButton.addEventListener("click", () => {
        if (this.partialEvaluator) {
          this.partialEvaluator.stopEvaluation();
        }
        this.hideAbstractEvaluationModal();
      });
    }
  }

  /**
   * Update progress in the abstract evaluation modal
   */
  private updateAbstractEvaluationProgress(message: string, progress: number): void {
    const progressBar = document.getElementById("abstract-progress-bar");
    const progressText = document.getElementById("abstract-progress-text");
    const progressPercentage = document.getElementById("abstract-progress-percentage");

    if (progressBar) {
      progressBar.style.width = `${progress}%`;
    }

    if (progressText) {
      progressText.textContent = message;
    }

    if (progressPercentage) {
      progressPercentage.textContent = `${Math.round(progress)}%`;
    }
  }

  /**
   * Hide the abstract evaluation modal
   */
  private hideAbstractEvaluationModal(): void {
    const modal = document.getElementById("abstract-evaluation-modal");
    if (modal) {
      modal.remove();
    }
  }

  /**
   * Show error message in abstract tab
   */
  private showAbstractError(message: string, isError = true): void {
    const errorEl = document.getElementById("abstract-error");
    const errorMessage = document.getElementById("abstract-error-message");

    if (errorEl && errorMessage) {
      const prefix = isError ? "Error: " : "";
      errorMessage.innerHTML = `<strong>${prefix}</strong>${message}`;
      if (isError) {
        errorEl.className =
          "rounded-lg border border-red-500 bg-red-100/50 p-4 " +
          "text-red-700 dark:bg-red-900/20 dark:text-red-300";
      } else {
        errorEl.className =
          "rounded-lg border border-green-500 bg-green-100/50 p-4 " +
          "text-green-700 dark:bg-green-900/20 dark:text-green-300";
      }
      errorEl.classList.remove("hidden");
    }
  }

  /**
   * Hide error message in abstract tab
   */
  private hideAbstractError(): void {
    const errorEl = document.getElementById("abstract-error");
    if (errorEl) {
      errorEl.classList.add("hidden");
    }
  }

  /**
   * Show completion notification for abstract evaluation
   */
  private showAbstractCompletionNotification(title: string): void {
    // Create notification element
    const notification = document.createElement("div");
    notification.className =
      "fixed top-4 right-4 z-50 rounded-lg bg-green-600 p-4 text-white shadow-lg dark:bg-green-700";

    notification.innerHTML = `
      <div class="flex items-center gap-3">
        <svg class="h-5 w-5 text-green-200" fill="currentColor" viewBox="0 0 20 20">
          <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd" />
        </svg>
        <div>
          <p class="font-medium">Abstract Evaluation Complete!</p>
          <p class="text-sm text-green-200 line-clamp-1">
            ${title.length > 50 ? title.substring(0, 50) + "..." : title}
          </p>
        </div>
      </div>
    `;

    document.body.appendChild(notification);

    // Remove after 4 seconds
    setTimeout(() => {
      if (document.body.contains(notification)) {
        notification.remove();
      }
    }, 4000);
  }
}

/**
 * Initialise the Paper Explorer application.
 */
async function initialiseApp(): Promise<void> {
  await waitForDOM();

  if (isMobileDevice()) {
    showMobileMessage();
    return;
  }

  // Get environment variables
  const jsonPath = import.meta.env.VITE_XP_DATA_PATH as string | undefined;
  const apiUrl = import.meta.env.VITE_API_URL as string | undefined;

  if (!jsonPath) {
    throw new Error("VITE_XP_DATA_PATH environment variable is required");
  }
  console.debug(`API URL: ${apiUrl}`);

  // Create services
  const jsonDataset = new JsonPaperDataset(jsonPath);
  const arxivService = apiUrl ? new ArxivPaperService(apiUrl) : null;
  const paperEvaluator = apiUrl ? new PaperEvaluator(apiUrl) : null;
  const partialEvaluator = apiUrl ? new PartialPaperEvaluator(apiUrl) : null;

  const explorer = new PaperExplorer(
    jsonDataset,
    arxivService,
    paperEvaluator,
    partialEvaluator,
  );
  await retryWithBackoff(() => explorer.initialize());

  // Add footer
  addFooter();
}

initialiseApp().catch(showInitError);
