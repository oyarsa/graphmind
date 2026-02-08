// @vitest-environment jsdom
import { describe, expect, it, vi, afterEach } from "vitest";

import {
  buildAbstractDetailPageContext,
  buildDetailPageContext,
  PaperChatWidget,
  PaperChatService,
  renderSimpleMarkdown,
} from "./chat";
import type { AbstractEvaluationResponse, GraphResult } from "./model";

afterEach(() => {
  vi.restoreAllMocks();
  document.body.innerHTML = "";
  localStorage.clear();
});

function makeGraphResult(): GraphResult {
  return {
    graph: {
      title: "Demo",
      abstract: "Graph abstract",
      entities: [
        { label: "Keyword A", type: "keyword", detail: null, excerpts: null },
        { label: "Method", type: "method", detail: null, excerpts: null },
      ],
      relationships: [],
      valid_status: "Valid",
      valid_status_all: ["Valid"],
    },
    paper: {
      id: "paper-1",
      title: "Demo Paper",
      year: 2024,
      authors: ["A. Author"],
      abstract: "A".repeat(1400),
      conference: "NeurIPS",
      rating: 4,
      y_pred: 4,
      y_true: 4,
      rationale_pred: "Rationale",
      rationale_true: "Human rationale",
      sections: [],
      references: null,
      approval: true,
      structured_evaluation: {
        paper_summary: "S".repeat(500),
        supporting_evidence: [{ text: "Support text", paper_title: "Support paper" }],
        contradictory_evidence: ["Contradictory item"],
        key_comparisons: ["Comparison 1"],
        conclusion: "C".repeat(500),
        label: 4,
        probability: 0.7,
        confidence: 0.8,
      },
      arxiv_id: "2401.00001",
    },
    related: [
      {
        abstract: "Related abstract",
        paper_id: "r1",
        polarity: "positive",
        score: 0.91,
        source: "semantic",
        summary: "R".repeat(500),
        title: "Related title",
        year: 2022,
        authors: ["B. Author"],
        venue: null,
        citation_count: null,
        reference_count: null,
        influential_citation_count: null,
        corpus_id: null,
        url: null,
        arxiv_id: null,
        contexts: null,
        background: null,
        target: "target",
      },
    ],
    terms: null,
    background: null,
    target: null,
  };
}

function makeAbstractEvaluation(): AbstractEvaluationResponse {
  return {
    id: "abs-1",
    title: "Abstract demo",
    abstract: "Abstract body",
    keywords: ["k1", "k2"],
    label: 3,
    paper_summary: "Summary",
    key_comparisons: ["Comp"],
    supporting_evidence: [{ text: "Support", paper_title: "P1", source: "semantic" }],
    contradictory_evidence: [
      { text: "Contradict", paper_title: "P2", source: "citations" },
    ],
    conclusion: "Conclusion",
    total_cost: 0.2,
    related: [
      {
        abstract: "Related abstract",
        paper_id: "r2",
        polarity: "negative",
        score: 0.4,
        source: "citations",
        summary: "Related summary",
        title: "Related title",
        year: 2020,
        authors: ["C. Author"],
        venue: null,
        citation_count: null,
        reference_count: null,
        influential_citation_count: null,
        corpus_id: null,
        url: null,
        arxiv_id: null,
        contexts: null,
        background: null,
        target: null,
      },
    ],
  };
}

describe("chat context builders", () => {
  it("builds detail context with expected shape", () => {
    const context = buildDetailPageContext(makeGraphResult()) as {
      paper: { abstract: string; title: string };
      evaluation: { paper_summary: string; label: number };
      keywords: string[];
      related_papers: unknown[];
    };

    expect(context.paper.title).toBe("Demo Paper");
    expect(context.paper.abstract).toBe("A".repeat(1400));
    expect(context.evaluation.label).toBe(4);
    expect(context.keywords).toEqual(["Keyword A"]);
    expect(context.related_papers).toHaveLength(1);
  });

  it("builds abstract detail context with expected shape", () => {
    const context = buildAbstractDetailPageContext(makeAbstractEvaluation()) as {
      paper: { title: string };
      evaluation: { label: number };
      keywords: string[];
    };

    expect(context.paper.title).toBe("Abstract demo");
    expect(context.evaluation.label).toBe(3);
    expect(context.keywords).toEqual(["k1", "k2"]);
  });
});

describe("PaperChatService", () => {
  it("posts chat payload and parses response", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ assistant_message: "hello", cost: 0.05 }),
    });
    vi.stubGlobal("fetch", fetchMock);

    const service = new PaperChatService("http://localhost:8000");
    const result = await service.sendMessage({
      messages: [{ role: "user", content: "Hi" }],
      page_context: { paper: { title: "Demo" } },
      llm_model: "gpt-4o-mini",
      page_type: "detail",
    });

    expect(result.assistant_message).toBe("hello");
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/mind/chat",
      expect.objectContaining({
        method: "POST",
      }),
    );
  });
});

describe("PaperChatWidget", () => {
  function createWidget(): PaperChatWidget {
    return new PaperChatWidget({
      baseUrl: "http://localhost:8000",
      pageType: "detail",
      conversationId: "paper-1",
      getPageContext: () => ({ paper: { title: "Demo" } }),
      getModel: () => "gpt-4o-mini",
    });
  }

  it("sends at most 20 messages even when local history is larger", async () => {
    const history = Array.from({ length: 20 }, (_, i) => ({
      role: (i % 2 === 0 ? "user" : "assistant") as "user" | "assistant",
      content: `history-${i}`,
    }));
    localStorage.setItem("paper-chat-history:detail:paper-1", JSON.stringify(history));

    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ assistant_message: "ok" }),
    });
    vi.stubGlobal("fetch", fetchMock);

    const widget = createWidget();
    const input = document.querySelector("textarea") as HTMLTextAreaElement;
    input.value = "latest question";
    await (widget as any).sendCurrentMessage();

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [, request] = fetchMock.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(request.body as string) as {
      messages: Array<{ role: string; content: string }>;
    };
    expect(body.messages).toHaveLength(20);
    expect(body.messages.at(-1)?.content).toBe("latest question");
  });

  it("discards late responses after switching conversations", async () => {
    let resolveFetch: ((value: unknown) => void) | null = null;
    const fetchMock = vi.fn().mockImplementation(
      () =>
        new Promise(resolve => {
          resolveFetch = resolve;
        }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const widget = createWidget();
    const input = document.querySelector("textarea") as HTMLTextAreaElement;
    input.value = "question for paper 1";
    const pending = (widget as any).sendCurrentMessage();

    widget.switchConversation("paper-2", () => ({ paper: { title: "Paper 2" } }));
    resolveFetch?.({
      ok: true,
      json: () => Promise.resolve({ assistant_message: "stale response" }),
    });
    await pending;

    const paper2History = localStorage.getItem("paper-chat-history:detail:paper-2");
    expect(paper2History).toBeNull();
    expect(document.body.textContent).not.toContain("stale response");
  });
});

describe("renderSimpleMarkdown", () => {
  it("renders bold and italic", () => {
    expect(renderSimpleMarkdown("**bold** and *italic*")).toContain(
      "<strong>bold</strong>",
    );
    expect(renderSimpleMarkdown("**bold** and *italic*")).toContain("<em>italic</em>");
  });

  it("renders inline code", () => {
    const html = renderSimpleMarkdown("use `foo()` here");
    expect(html).toContain("<code");
    expect(html).toContain("foo()");
  });

  it("renders code blocks", () => {
    const html = renderSimpleMarkdown("```\nprint('hi')\n```");
    expect(html).toContain("<pre");
    expect(html).toContain("print('hi')");
  });

  it("renders bullet lists", () => {
    const html = renderSimpleMarkdown("- one\n- two");
    expect(html).toContain("<ul");
    expect(html).toContain("<li>");
    expect(html).toContain("one");
    expect(html).toContain("two");
  });

  it("escapes HTML to prevent XSS", () => {
    const html = renderSimpleMarkdown('<script>alert("xss")</script>');
    expect(html).not.toContain("<script>");
    expect(html).toBe("");
  });
});
