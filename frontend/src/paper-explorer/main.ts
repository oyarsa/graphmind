import { z } from "zod/v4";

import {
  retryWithBackoff,
  waitForDOM,
  showInitError,
  isMobileDevice,
  showMobileMessage,
  cleanKeyword,
} from "../util";
import { GraphResult, PaperSearchResults, PaperSearchItem } from "./model";
import { renderLatex, getArxivUrl, formatConferenceName } from "./helpers";
import {
  JsonPaperDataset,
  ArxivPaperService,
  PaperEvaluator,
  PaperEvaluatorMulti,
  AbstractPaperEvaluator,
  AbstractEvaluationParams,
} from "./services";
import { addFooter } from "../footer";
import { loadSettings } from "./settings";
import {
  showEvaluationModal,
  updateEvaluationProgress,
  showEvaluationError,
  hideEvaluationModal,
  showEvaluationSettingsModal,
  hideEvaluationSettingsModal,
  setupSettingsModal,
  setupHelpModal,
  requestNotificationPermission,
  showCompletionNotification,
} from "./modals";
import {
  type ArxivMode,
  type CachedPaperSearchItem,
  getCachedPapers,
  convertCachedPapersToSearchResults,
  getCacheKeys,
  clearCache,
  findCachedPaperByArxivId,
  storePaperInCache,
} from "./cache";
import {
  setupAbstractTab,
  loadPreviousAbstractEvaluations,
  showAbstractEvaluationModal,
  updateAbstractEvaluationProgress,
  hideAbstractEvaluationModal,
  showAbstractError,
  showAbstractCompletionNotification,
  cacheAbstractEvaluation,
} from "./abstract";

class PaperExplorer {
  private allPapers: GraphResult[] = [];
  private filteredPapers: GraphResult[] = [];
  private lastSelectedArxivItem: PaperSearchItem | null = null;
  private currentArxivMode: ArxivMode = "default";

  constructor(
    private jsonDataset: JsonPaperDataset,
    private arxivService: ArxivPaperService | null = null,
    private paperEvaluator: PaperEvaluator | null = null,
    private paperEvaluatorMulti: PaperEvaluatorMulti | null = null,
    private abstractEvaluator: AbstractPaperEvaluator | null = null,
  ) {}

  async initialize(): Promise<void> {
    this.setupTabs();
    if (this.arxivService) {
      this.setupArxivModeToggle();
      this.setupArxivSearchBar();
      this.setupClearCacheButton();
      this.setupEvaluationModal();
    }
    if (this.abstractEvaluator) {
      setupAbstractTab(this.abstractEvaluator, params => this.evaluateAbstract(params));
    }
    setupHelpModal();
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

      try {
        localStorage.setItem("papers-dataset", JSON.stringify(validatedData));
      } catch (error) {
        console.warn("Failed to cache papers dataset in localStorage:", error);
      }

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
      paperLink.href = `/graphmind/pages/detail.html?id=${encodedId}`;
      paperLink.className =
        "block rounded-lg bg-gray-100/50 dark:bg-gray-900/50 border border-gray-300" +
        " dark:border-gray-700 p-6 transition-all duration-200 hover:border-teal-500/50" +
        " hover:-translate-y-1 no-underline";

      const keywords = graphResult.graph.entities
        .filter(e => e.type === "keyword")
        .map(e => cleanKeyword(e.label))
        .slice(0, 5);

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

    if (!jsonTab || !arxivTab || !abstractTab) return;

    jsonTab.addEventListener("click", () => this.switchTab("json"));
    arxivTab.addEventListener("click", () => this.switchTab("arxiv"));
    abstractTab.addEventListener("click", () => this.switchTab("abstract"));
  }

  private switchTab(tab: "json" | "arxiv" | "abstract"): void {
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

    const activeClasses =
      "tab-button active cursor-pointer rounded-md bg-white px-4 py-2 text-sm font-semibold" +
      " text-teal-600 shadow-sm transition-all duration-200 dark:bg-gray-700 dark:text-teal-400";
    const inactiveClasses =
      "tab-button cursor-pointer rounded-md px-4 py-2 text-sm font-semibold text-gray-600" +
      " transition-all duration-200 hover:text-gray-800 dark:text-gray-400 dark:hover:text-gray-200";

    jsonTab.className = inactiveClasses;
    arxivTab.className = inactiveClasses;
    abstractTab.className = inactiveClasses;

    jsonContent.classList.add("hidden");
    arxivContent.classList.add("hidden");
    abstractContent.classList.add("hidden");

    if (tab === "json") {
      jsonTab.className = activeClasses;
      jsonContent.classList.remove("hidden");
    } else if (tab === "arxiv") {
      arxivTab.className = activeClasses;
      arxivContent.classList.remove("hidden");
    } else {
      abstractTab.className = activeClasses;
      abstractContent.classList.remove("hidden");
      loadPreviousAbstractEvaluations();
    }
  }

  private setupArxivModeToggle(): void {
    const defaultRadio = document.getElementById(
      "arxiv-mode-default",
    ) as HTMLInputElement | null;
    const multiRadio = document.getElementById(
      "arxiv-mode-multi",
    ) as HTMLInputElement | null;
    const defaultLabel = document.querySelector<HTMLElement>(
      'label[for="arxiv-mode-default"]',
    );
    const multiLabel = document.querySelector<HTMLElement>(
      'label[for="arxiv-mode-multi"]',
    );

    if (!defaultRadio || !multiRadio || !defaultLabel || !multiLabel) {
      console.warn("ArXiv mode toggle elements not found");
      return;
    }

    const updateToggleStyle = (mode: ArxivMode) => {
      const activeClasses =
        "arxiv-mode-label cursor-pointer rounded-md px-4 py-2 text-sm font-semibold " +
        "bg-white text-teal-600 shadow-sm dark:bg-gray-700 dark:text-teal-400";
      const inactiveClasses =
        "arxiv-mode-label cursor-pointer rounded-md px-4 py-2 text-sm font-semibold " +
        "text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200";

      if (mode === "default") {
        defaultLabel.className = activeClasses;
        multiLabel.className = inactiveClasses;
      } else {
        defaultLabel.className = inactiveClasses;
        multiLabel.className = activeClasses;
      }
    };

    const switchMode = (mode: ArxivMode) => {
      this.currentArxivMode = mode;
      updateToggleStyle(mode);

      const modeDescription = document.getElementById("arxiv-mode-description");
      if (modeDescription) {
        if (mode === "multi") {
          modeDescription.textContent =
            "Multi-perspective evaluation with separate analysis from different viewpoints.";
        } else {
          modeDescription.textContent =
            "Comprehensive analysis with novelty assessment and related papers discovery.";
        }
      }

      const searchInput = document.getElementById(
        "arxiv-search-input",
      ) as HTMLInputElement | null;
      if (searchInput) {
        searchInput.value = "";
      }

      this.displayCachedPapersIfEmpty();

      const resultCount = document.getElementById("arxiv-result-count");
      if (resultCount) {
        resultCount.textContent = "";
      }
    };

    defaultRadio.addEventListener("change", () => {
      if (defaultRadio.checked) {
        switchMode("default");
      }
    });

    multiRadio.addEventListener("change", () => {
      if (multiRadio.checked) {
        switchMode("multi");
      }
    });

    defaultRadio.checked = true;
    multiRadio.checked = false;
    updateToggleStyle("default");
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
              arXiv
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

    formElement.addEventListener("submit", event => {
      event.preventDefault();
      performSearch();
    });

    inputElement.addEventListener("input", updateClearButton);

    clearElement.addEventListener("click", () => {
      inputElement.value = "";
      clearElement.style.display = "none";
      inputElement.focus();
      this.displayCachedPapersIfEmpty();
    });

    updateClearButton();
    this.displayCachedPapersIfEmpty();
  }

  private displayCachedPapersIfEmpty(): void {
    const searchInput = document.getElementById("arxiv-search-input");
    if (!searchInput || (searchInput as HTMLInputElement).value.trim()) return;

    const cachedPapers = getCachedPapers(this.currentArxivMode);
    const papersContainer = document.getElementById("arxiv-papers-container");
    const resultCount = document.getElementById("arxiv-result-count");

    if (cachedPapers.length > 0) {
      if (papersContainer) papersContainer.classList.remove("hidden");
      const searchResults = convertCachedPapersToSearchResults(cachedPapers);
      this.displayArxivPapers(searchResults as PaperSearchResults);

      if (resultCount) {
        resultCount.textContent =
          `${cachedPapers.length}` +
          ` previous evaluation${cachedPapers.length === 1 ? "" : "s"}`;
      }
    } else {
      if (papersContainer) {
        papersContainer.innerHTML = "";
        papersContainer.classList.add("hidden");
      }
      if (resultCount) {
        resultCount.textContent = "";
      }
    }
  }

  private handleArxivPaperSelection(item: PaperSearchItem): void {
    const cachedItem = item as CachedPaperSearchItem;
    if (cachedItem._isCached && cachedItem._cachedPaperId) {
      const encodedId = encodeURIComponent(cachedItem._cachedPaperId);
      const detailPage =
        this.currentArxivMode === "multi"
          ? "/graphmind/pages/detail-multi.html"
          : "/graphmind/pages/detail.html";
      window.location.href = `${detailPage}?id=${encodedId}`;
      return;
    }

    if (!this.arxivService) {
      console.error("ArXiv service not available");
      return;
    }

    this.lastSelectedArxivItem = item;

    const modeText =
      this.currentArxivMode === "multi" ? "multi-perspective" : "default";
    console.log(`[Cache] Checking ${modeText} cache for arXiv paper: ${item.arxiv_id}`);

    const cachedPaper = findCachedPaperByArxivId(item.arxiv_id, this.currentArxivMode);
    if (cachedPaper) {
      const encodedId = encodeURIComponent(cachedPaper.paper.id);
      const detailPage =
        this.currentArxivMode === "multi"
          ? "/graphmind/pages/detail-multi.html"
          : "/graphmind/pages/detail.html";
      window.location.href = `${detailPage}?id=${encodedId}`;
      return;
    }

    showEvaluationSettingsModal(item.title);
  }

  private setupClearCacheButton(): void {
    const clearButton = document.getElementById("clear-cache");
    if (!clearButton) return;

    clearButton.addEventListener("click", () => {
      const cacheKeys = getCacheKeys(this.currentArxivMode);

      if (cacheKeys.length === 0) {
        const modeText =
          this.currentArxivMode === "multi" ? "multi-perspective" : "default";
        alert(`No previous ${modeText} results found.`);
        return;
      }

      const modeText =
        this.currentArxivMode === "multi" ? "multi-perspective" : "default";
      if (
        confirm(
          `Clear ${cacheKeys.length} previous ${modeText} results? This cannot be undone.`,
        )
      ) {
        clearCache(this.currentArxivMode);
        const searchInput = document.getElementById("arxiv-search-input");
        if (searchInput && !(searchInput as HTMLInputElement).value.trim()) {
          this.displayCachedPapersIfEmpty();
        }
      }
    });
  }

  private setupEvaluationModal(): void {
    const modal = document.getElementById("evaluation-settings-modal");
    const cancelButton = document.getElementById("cancel-evaluation");
    const startButton = document.getElementById("start-evaluation");
    const paperTitleEl = document.getElementById("evaluation-paper-title");

    if (!modal || !cancelButton || !startButton || !paperTitleEl) {
      console.warn("Evaluation modal elements not found");
      return;
    }

    cancelButton.addEventListener("click", () => hideEvaluationSettingsModal());

    startButton.addEventListener("click", () => {
      void this.handleEvaluationSubmit();
    });

    modal.addEventListener("click", e => {
      if (e.target === modal) {
        hideEvaluationSettingsModal();
      }
    });

    document.addEventListener("keydown", e => {
      if (e.key === "Escape" && !modal.classList.contains("hidden")) {
        hideEvaluationSettingsModal();
      }
    });

    setupSettingsModal();
  }

  private async handleEvaluationSubmit(): Promise<void> {
    if (!this.lastSelectedArxivItem) {
      console.error("No selected item available");
      return;
    }

    const currentEvaluator =
      this.currentArxivMode === "multi"
        ? this.paperEvaluatorMulti
        : this.paperEvaluator;
    if (!currentEvaluator) {
      console.error(`${this.currentArxivMode} evaluator not available`);
      return;
    }

    const settings = loadSettings();

    const params = {
      id: this.lastSelectedArxivItem.arxiv_id,
      title: this.lastSelectedArxivItem.title,
      k_refs: settings.k_refs,
      recommendations: settings.recommendations,
      related: settings.related,
      llm_model: settings.llm_model,
      filter_by_date: settings.filter_by_date,
      seed: 0,
    };

    hideEvaluationSettingsModal();
    await requestNotificationPermission();
    showEvaluationModal(params.title);

    const cancelButton = document.getElementById("cancel-arxiv-evaluation");
    if (cancelButton) {
      cancelButton.addEventListener("click", () => {
        if (this.paperEvaluator) {
          this.paperEvaluator.stopEvaluation();
        }
        hideEvaluationModal();
      });
    }

    try {
      currentEvaluator.onConnected = (message: string) => {
        console.log("Connected:", message);
        updateEvaluationProgress(message, 0);
      };

      currentEvaluator.onProgress = (message: string, progress: number) => {
        console.log(`Progress: ${message} (${progress.toFixed(1)}%)`);
        updateEvaluationProgress(message, progress);
      };

      currentEvaluator.onError = (error: string) => {
        console.error("Evaluation error:", error);
        showEvaluationError(
          error,
          () => {
            hideEvaluationModal();
            if (this.lastSelectedArxivItem) {
              this.handleArxivPaperSelection(this.lastSelectedArxivItem);
            }
          },
          () => hideEvaluationModal(),
        );
      };

      currentEvaluator.onConnectionError = (event: Event) => {
        console.error("Connection error:", event);
        showEvaluationError(
          "Connection lost. Please try again.",
          () => {
            hideEvaluationModal();
            if (this.lastSelectedArxivItem) {
              this.handleArxivPaperSelection(this.lastSelectedArxivItem);
            }
          },
          () => hideEvaluationModal(),
        );
      };

      const evalResult = await currentEvaluator.startEvaluation(params);

      updateEvaluationProgress("Evaluation complete", 100);

      const paperId = evalResult.result.paper.id;
      storePaperInCache(paperId, evalResult.result, this.currentArxivMode);

      showCompletionNotification(params.title);

      const encodedId = encodeURIComponent(paperId);
      const detailPage =
        this.currentArxivMode === "multi"
          ? "/graphmind/pages/detail-multi.html"
          : "/graphmind/pages/detail.html";
      window.location.href = `${detailPage}?id=${encodedId}`;
    } catch (error) {
      if (error instanceof Error && error.message.includes("cancelled by user")) {
        console.log("Evaluation cancelled by user");
        return;
      }

      console.error("Error evaluating paper:", error);
      showEvaluationError(
        "Failed to evaluate paper. Please try again.",
        () => {
          hideEvaluationModal();
          if (this.lastSelectedArxivItem) {
            this.handleArxivPaperSelection(this.lastSelectedArxivItem);
          }
        },
        () => hideEvaluationModal(),
      );
    }
  }

  private async evaluateAbstract(params: AbstractEvaluationParams): Promise<void> {
    if (!this.abstractEvaluator) {
      showAbstractError("Abstract evaluation service not available.");
      return;
    }

    try {
      showAbstractEvaluationModal(params.title);

      this.abstractEvaluator.onConnected = message => {
        updateAbstractEvaluationProgress(message, 0);
      };

      this.abstractEvaluator.onProgress = (message, progress) => {
        updateAbstractEvaluationProgress(message, progress);
      };

      this.abstractEvaluator.onError = error => {
        hideAbstractEvaluationModal();
        showAbstractError(`Evaluation failed: ${error}`);
      };

      this.abstractEvaluator.onConnectionError = () => {
        hideAbstractEvaluationModal();
        showAbstractError("Connection lost during evaluation. Please try again.");
      };

      const cancelButton = document.getElementById("cancel-abstract-evaluation");
      if (cancelButton) {
        cancelButton.addEventListener("click", () => {
          if (this.abstractEvaluator) {
            this.abstractEvaluator.stopEvaluation();
          }
          hideAbstractEvaluationModal();
        });
      }

      const result = await this.abstractEvaluator.startEvaluation(params);

      updateAbstractEvaluationProgress("Evaluation complete", 100);
      hideAbstractEvaluationModal();

      cacheAbstractEvaluation(result);
      showAbstractCompletionNotification(params.title);

      window.location.href = `/graphmind/pages/abstract-detail.html?id=${result.id}`;
    } catch (error) {
      if (error instanceof Error && error.message.includes("cancelled by user")) {
        console.log("Abstract evaluation cancelled by user");
        return;
      }

      hideAbstractEvaluationModal();
      console.error("Error evaluating abstract:", error);
      showAbstractError("Failed to evaluate abstract. Please try again.");
    }
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

  const jsonPath = import.meta.env.VITE_XP_DATA_PATH as string | undefined;
  const apiUrl = import.meta.env.VITE_API_URL as string | undefined;

  if (!jsonPath) {
    throw new Error("VITE_XP_DATA_PATH environment variable is required");
  }
  console.debug(`API URL: ${apiUrl}`);

  const jsonDataset = new JsonPaperDataset(jsonPath);
  const arxivService = apiUrl ? new ArxivPaperService(apiUrl) : null;
  const paperEvaluator = apiUrl ? new PaperEvaluator(apiUrl) : null;
  const paperEvaluatorMulti = apiUrl ? new PaperEvaluatorMulti(apiUrl) : null;
  const abstractEvaluator = apiUrl ? new AbstractPaperEvaluator(apiUrl) : null;

  const explorer = new PaperExplorer(
    jsonDataset,
    arxivService,
    paperEvaluator,
    paperEvaluatorMulti,
    abstractEvaluator,
  );
  await retryWithBackoff(() => explorer.initialize());

  addFooter();
}

initialiseApp().catch(showInitError);
