import { describe, expect, it, vi, afterEach } from "vitest";

import {
  buildAbstractDetailPageContext,
  buildDetailPageContext,
  PaperChatService,
} from "./chat";
import type { AbstractEvaluationResponse, GraphResult } from "./model";

afterEach(() => {
  vi.restoreAllMocks();
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
  it("builds bounded detail context", () => {
    const context = buildDetailPageContext(makeGraphResult()) as {
      paper: { abstract: string };
      evaluation: { paper_summary: string };
      keywords: string[];
      related_papers: unknown[];
    };

    expect(context.paper.abstract.length).toBeLessThanOrEqual(1203);
    expect(context.evaluation.paper_summary.length).toBeLessThanOrEqual(453);
    expect(context.keywords).toEqual(["Keyword A"]);
    expect(context.related_papers.length).toBe(1);
  });

  it("builds abstract detail context", () => {
    const context = buildAbstractDetailPageContext(makeAbstractEvaluation()) as {
      paper: { title: string };
      evaluation: { label: number };
      keywords: string[];
    };

    expect(context.paper.title).toContain("Abstract demo");
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
