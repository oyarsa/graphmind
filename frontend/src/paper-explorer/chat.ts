import type { AbstractEvaluationResponse, GraphResult } from "./model";

export type ChatRole = "user" | "assistant";

export interface ChatMessage {
  role: ChatRole;
  content: string;
}

export type ChatPageType = "detail" | "abstract-detail";

interface ChatResponse {
  assistant_message: string;
  cost?: number;
}

interface ChatRequestBody {
  messages: ChatMessage[];
  page_context: Record<string, unknown>;
  llm_model: "gpt-4o" | "gpt-4o-mini" | "gemini-2.0-flash";
  page_type: ChatPageType;
}

const CHAT_MIN_WIDTH_PX = 320;
const CHAT_MIN_HEIGHT_PX = 384;

/** Build chat context from a full detail page. Backend handles all truncation. */
export function buildDetailPageContext(
  graphResult: GraphResult,
): Record<string, unknown> {
  const { paper, graph, related } = graphResult;
  const keywords = graph.entities
    .filter(entity => entity.type === "keyword")
    .map(entity => entity.label);

  const evaluation = paper.structured_evaluation;

  return {
    paper: {
      title: paper.title,
      abstract: paper.abstract,
      year: paper.year,
      conference: paper.conference,
      arxiv_id: paper.arxiv_id,
    },
    keywords,
    evaluation: evaluation
      ? {
          label: evaluation.label,
          paper_summary: evaluation.paper_summary,
          conclusion: evaluation.conclusion,
          key_comparisons: evaluation.key_comparisons,
          supporting_evidence: evaluation.supporting_evidence,
          contradictory_evidence: evaluation.contradictory_evidence,
        }
      : null,
    related_papers: related.map(rp => ({
      title: rp.title,
      summary: rp.summary,
      source: rp.source,
      score: rp.score,
      year: rp.year,
    })),
  };
}

/** Build chat context from an abstract evaluation page. Backend handles all truncation. */
export function buildAbstractDetailPageContext(
  evaluation: AbstractEvaluationResponse,
): Record<string, unknown> {
  return {
    paper: {
      title: evaluation.title,
      abstract: evaluation.abstract,
    },
    keywords: evaluation.keywords,
    evaluation: {
      label: evaluation.label,
      paper_summary: evaluation.paper_summary,
      conclusion: evaluation.conclusion,
      key_comparisons: evaluation.key_comparisons,
      supporting_evidence: evaluation.supporting_evidence,
      contradictory_evidence: evaluation.contradictory_evidence,
    },
    related_papers: evaluation.related.map(rp => ({
      title: rp.title,
      summary: rp.summary,
      source: rp.source,
      score: rp.score,
      year: rp.year,
    })),
  };
}

export class PaperChatService {
  constructor(private baseUrl: string) {}

  async sendMessage(
    request: ChatRequestBody,
    signal?: AbortSignal,
  ): Promise<ChatResponse> {
    const response = await fetch(`${this.baseUrl}/mind/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
      signal,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText || `Chat request failed: ${response.status}`);
    }

    return (await response.json()) as ChatResponse;
  }
}

interface PaperChatWidgetOptions {
  baseUrl: string;
  pageType: ChatPageType;
  conversationId: string;
  getPageContext: () => Record<string, unknown>;
  getModel: () => "gpt-4o" | "gpt-4o-mini" | "gemini-2.0-flash";
}

export class PaperChatWidget {
  private readonly service: PaperChatService;
  private readonly messages: ChatMessage[] = [];
  private conversationStorageKey: string;
  private readonly panel: HTMLDivElement;
  private readonly transcript: HTMLDivElement;
  private readonly input: HTMLTextAreaElement;
  private readonly sendButton: HTMLButtonElement;
  private readonly clearButton: HTMLButtonElement;
  private readonly closeButton: HTMLButtonElement;
  private readonly errorEl: HTMLDivElement;
  private getPageContext: () => Record<string, unknown>;
  private isSending = false;
  private abortController: AbortController | null = null;

  constructor(private options: PaperChatWidgetOptions) {
    this.service = new PaperChatService(options.baseUrl);
    this.conversationStorageKey = this.buildStorageKey(
      options.pageType,
      options.conversationId,
    );
    this.getPageContext = options.getPageContext;

    const existing = document.getElementById("paper-chat-widget-root");
    existing?.remove();

    const root = document.createElement("div");
    root.id = "paper-chat-widget-root";
    root.className = "fixed right-4 bottom-4 z-50";

    const fab = document.createElement("button");
    fab.type = "button";
    fab.className =
      "rounded-full bg-teal-600 px-4 py-3 text-sm font-semibold text-white shadow-lg transition-colors hover:bg-teal-700 dark:bg-teal-500 dark:hover:bg-teal-600";
    fab.textContent = "Paper chat";

    this.panel = document.createElement("div");
    this.panel.className =
      "relative mt-3 hidden h-[34rem] w-[min(32rem,calc(100vw-2rem))] min-h-[24rem] min-w-[20rem] max-h-[80vh] max-w-[90vw] overflow-hidden flex flex-col rounded-xl border border-gray-300 bg-white shadow-2xl dark:border-gray-700 dark:bg-gray-900";

    const resizeCornerHandle = document.createElement("div");
    resizeCornerHandle.className =
      "group absolute top-0 left-0 z-20 h-5 w-5 cursor-nwse-resize touch-none";
    resizeCornerHandle.title = "Resize chat";
    const resizeCornerGrip = document.createElement("div");
    resizeCornerGrip.className =
      "h-full w-full bg-[linear-gradient(135deg,rgba(0,0,0,0.12)_50%,transparent_50%)] opacity-70 transition-opacity group-hover:opacity-100 dark:bg-[linear-gradient(135deg,rgba(255,255,255,0.18)_50%,transparent_50%)]";
    resizeCornerHandle.append(resizeCornerGrip);
    this.panel.append(resizeCornerHandle);

    const resizeTopHandle = document.createElement("div");
    resizeTopHandle.className =
      "absolute top-0 left-5 right-0 z-10 h-2 cursor-ns-resize touch-none";
    resizeTopHandle.title = "Resize chat height";
    this.panel.append(resizeTopHandle);

    const resizeLeftHandle = document.createElement("div");
    resizeLeftHandle.className =
      "absolute top-5 bottom-0 left-0 z-10 w-2 cursor-ew-resize touch-none";
    resizeLeftHandle.title = "Resize chat width";
    this.panel.append(resizeLeftHandle);

    const header = document.createElement("div");
    header.className =
      "flex items-center justify-between border-b border-gray-200 px-4 py-3 dark:border-gray-700";

    const title = document.createElement("div");
    title.className = "text-sm font-semibold text-gray-900 dark:text-gray-100";
    title.textContent = "Paper chat";

    const headerActions = document.createElement("div");
    headerActions.className = "flex items-center gap-2";

    this.clearButton = document.createElement("button");
    this.clearButton.type = "button";
    this.clearButton.className =
      "rounded border border-gray-300 px-2 py-1 text-xs font-medium text-gray-700 transition-colors hover:bg-gray-100 dark:border-gray-600 dark:text-gray-200 dark:hover:bg-gray-800";
    this.clearButton.textContent = "Clear";

    this.closeButton = document.createElement("button");
    this.closeButton.type = "button";
    this.closeButton.className =
      "rounded border border-red-700 bg-red-600 px-2 py-1 text-xs font-medium text-white transition-colors hover:bg-red-700 dark:border-red-600 dark:bg-red-500 dark:hover:bg-red-600";
    this.closeButton.textContent = "X";
    this.closeButton.title = "Close chat";

    this.transcript = document.createElement("div");
    this.transcript.className = "flex-1 space-y-3 overflow-y-auto px-4 py-3";

    this.errorEl = document.createElement("div");
    this.errorEl.className = "hidden px-4 pb-2 text-xs text-red-600 dark:text-red-400";

    const inputWrap = document.createElement("div");
    inputWrap.className = "border-t border-gray-200 px-3 py-3 dark:border-gray-700";

    this.input = document.createElement("textarea");
    this.input.rows = 2;
    this.input.maxLength = 2000;
    this.input.placeholder = "Ask about this paper...";
    this.input.className =
      "w-full resize-none rounded-md border border-gray-300 bg-white px-3 py-2 text-sm text-gray-900 outline-none focus:border-teal-500 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-100";

    this.sendButton = document.createElement("button");
    this.sendButton.type = "button";
    this.sendButton.className =
      "mt-2 w-full rounded-md bg-teal-600 px-3 py-2 text-sm font-semibold text-white transition-colors hover:bg-teal-700 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-teal-500 dark:hover:bg-teal-600";
    this.sendButton.textContent = "Send";

    inputWrap.append(this.input, this.sendButton);
    headerActions.append(this.clearButton, this.closeButton);
    header.append(title, headerActions);
    this.panel.append(header, this.transcript, this.errorEl, inputWrap);
    root.append(fab, this.panel);
    document.body.append(root);

    fab.addEventListener("click", () => {
      this.panel.classList.toggle("hidden");
      fab.classList.toggle("hidden", !this.panel.classList.contains("hidden"));
      if (!this.panel.classList.contains("hidden")) {
        this.input.focus();
      }
    });

    this.sendButton.addEventListener("click", () => {
      if (this.isSending) {
        this.cancelPending();
      } else {
        void this.sendCurrentMessage();
      }
    });

    this.input.addEventListener("keydown", event => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        void this.sendCurrentMessage();
      }
    });

    this.clearButton.addEventListener("click", () => {
      this.clearConversation();
    });

    this.closeButton.addEventListener("click", () => {
      this.panel.classList.add("hidden");
      fab.classList.remove("hidden");
    });

    this.initialiseResizeHandle(resizeCornerHandle, "corner");
    this.initialiseResizeHandle(resizeTopHandle, "top");
    this.initialiseResizeHandle(resizeLeftHandle, "left");
    this.restoreConversation();
  }

  switchConversation(
    conversationId: string,
    getPageContext: () => Record<string, unknown>,
  ): void {
    this.cancelPending();
    this.getPageContext = getPageContext;
    const newKey = this.buildStorageKey(this.options.pageType, conversationId);
    if (newKey === this.conversationStorageKey) return;
    this.conversationStorageKey = newKey;
    this.messages.length = 0;
    this.transcript.innerHTML = "";
    this.errorEl.classList.add("hidden");
    this.restoreConversation();
  }

  private async sendCurrentMessage(): Promise<void> {
    if (this.isSending) return;

    const text = this.input.value.trim();
    if (!text) return;

    this.errorEl.classList.add("hidden");
    const userMessage: ChatMessage = { role: "user", content: text };
    this.messages.push(userMessage);
    this.renderMessage(userMessage);
    this.persistConversation();

    this.input.value = "";
    this.abortController = new AbortController();
    this.setSending(true);

    try {
      const response = await this.service.sendMessage(
        {
          messages: this.messages,
          page_context: this.getPageContext(),
          llm_model: this.options.getModel(),
          page_type: this.options.pageType,
        },
        this.abortController.signal,
      );

      const assistant: ChatMessage = {
        role: "assistant",
        content: response.assistant_message,
      };
      this.messages.push(assistant);
      this.renderMessage(assistant);
      this.persistConversation();
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        // User cancelled — no error to show.
      } else {
        this.errorEl.textContent =
          error instanceof Error
            ? `Chat failed: ${error.message}`
            : "Chat failed. Please try again.";
        this.errorEl.classList.remove("hidden");
      }
    } finally {
      this.abortController = null;
      this.setSending(false);
    }
  }

  private cancelPending(): void {
    if (this.abortController) {
      this.abortController.abort();
      this.abortController = null;
    }
  }

  private setSending(sending: boolean): void {
    this.isSending = sending;
    this.sendButton.disabled = false;
    this.input.disabled = sending;
    this.clearButton.disabled = sending;
    this.sendButton.textContent = sending ? "Cancel" : "Send";
  }

  private renderMessage(message: ChatMessage): void {
    const row = document.createElement("div");
    row.className = message.role === "user" ? "flex justify-end" : "flex justify-start";

    const bubble = document.createElement("div");
    bubble.className =
      message.role === "user"
        ? "max-w-[85%] rounded-lg bg-teal-600 px-3 py-2 text-sm text-white"
        : "max-w-[85%] rounded-lg border border-gray-300 bg-gray-50 px-3 py-2 text-sm text-gray-900 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-100";
    bubble.textContent = message.content;

    row.append(bubble);
    this.transcript.append(row);
    this.transcript.scrollTop = this.transcript.scrollHeight;
  }

  private buildStorageKey(pageType: ChatPageType, conversationId: string): string {
    return `paper-chat-history:${pageType}:${conversationId}`;
  }

  private persistConversation(): void {
    try {
      localStorage.setItem(this.conversationStorageKey, JSON.stringify(this.messages));
    } catch {
      // Ignore storage issues so chat remains usable.
    }
  }

  private restoreConversation(): void {
    let raw: string | null = null;
    try {
      raw = localStorage.getItem(this.conversationStorageKey);
    } catch {
      return;
    }
    if (!raw) return;

    try {
      const parsed = JSON.parse(raw) as unknown;
      if (!Array.isArray(parsed)) return;
      for (const item of parsed) {
        if (typeof item !== "object" || item === null) continue;
        const record = item as Record<string, unknown>;
        const role = record.role;
        const content = record.content;
        if ((role === "user" || role === "assistant") && typeof content === "string") {
          const message: ChatMessage = {
            role,
            content,
          };
          this.messages.push(message);
          this.renderMessage(message);
        }
      }
    } catch {
      // Ignore invalid cached history.
    }
  }

  private clearConversation(): void {
    this.messages.length = 0;
    this.transcript.innerHTML = "";
    this.errorEl.classList.add("hidden");
    try {
      localStorage.removeItem(this.conversationStorageKey);
    } catch {
      // Ignore storage issues so chat remains usable.
    }
    this.input.focus();
  }

  private initialiseResizeHandle(
    handle: HTMLDivElement,
    direction: "corner" | "top" | "left",
  ): void {
    handle.addEventListener("pointerdown", event => {
      if (event.button !== 0) return;
      event.preventDefault();

      const startX = event.clientX;
      const startY = event.clientY;
      const startWidth = this.panel.offsetWidth;
      const startHeight = this.panel.offsetHeight;

      const onPointerMove = (moveEvent: PointerEvent): void => {
        const deltaX = moveEvent.clientX - startX;
        const deltaY = moveEvent.clientY - startY;
        const maxWidth = Math.max(CHAT_MIN_WIDTH_PX, window.innerWidth * 0.9);
        const maxHeight = Math.max(CHAT_MIN_HEIGHT_PX, window.innerHeight * 0.8);
        if (direction === "corner" || direction === "left") {
          const width = this.clamp(startWidth - deltaX, CHAT_MIN_WIDTH_PX, maxWidth);
          this.panel.style.width = `${width}px`;
        }
        if (direction === "corner" || direction === "top") {
          const height = this.clamp(
            startHeight - deltaY,
            CHAT_MIN_HEIGHT_PX,
            maxHeight,
          );
          this.panel.style.height = `${height}px`;
        }
      };

      const onPointerUp = (): void => {
        document.removeEventListener("pointermove", onPointerMove);
        document.removeEventListener("pointerup", onPointerUp);
      };

      document.addEventListener("pointermove", onPointerMove);
      document.addEventListener("pointerup", onPointerUp);
    });
  }

  private clamp(value: number, min: number, max: number): number {
    return Math.min(Math.max(value, min), max);
  }
}
