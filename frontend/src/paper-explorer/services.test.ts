import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { type EvaluationParams } from "./model";
import { ArxivPaperService, PaperEvaluator, PaperEvaluatorMulti } from "./services";

class MockEventSource {
  static createdUrls: string[] = [];
  readonly url: string;
  onerror: ((event: Event) => unknown) | null = null;

  constructor(url: string) {
    this.url = url;
    MockEventSource.createdUrls.push(url);
  }

  addEventListener(): void {
    // No-op for URL construction tests.
  }

  close(): void {
    // No-op for URL construction tests.
  }
}

const originalEventSource = globalThis.EventSource;

function installMockEventSource(): void {
  Object.defineProperty(globalThis, "EventSource", {
    value: MockEventSource as unknown as typeof EventSource,
    configurable: true,
    writable: true,
  });
}

function restoreEventSource(): void {
  Object.defineProperty(globalThis, "EventSource", {
    value: originalEventSource,
    configurable: true,
    writable: true,
  });
}

describe("SSE evaluation URL encoding", () => {
  beforeEach(() => {
    MockEventSource.createdUrls = [];
    installMockEventSource();
  });

  afterEach(() => {
    restoreEventSource();
  });

  it("does not double-encode title in /mind/evaluate URL", async () => {
    const evaluator = new PaperEvaluator("http://localhost:8000");
    const params: EvaluationParams = {
      id: "1706.03762",
      title: "Attention Is All You Need",
      k_refs: 20,
      recommendations: 30,
      related: 5,
      llm_model: "gpt-4o-mini",
      filter_by_date: true,
      seed: 0,
    };

    const result = evaluator.startEvaluation(params);
    const url = MockEventSource.createdUrls[0];

    expect(url).toContain("/mind/evaluate?");
    expect(url).toContain("title=Attention+Is+All+You+Need");
    expect(url).not.toContain("%2520");
    expect(url).not.toContain("%2B");

    evaluator.stopEvaluation();
    await expect(result).rejects.toThrow("Evaluation cancelled by user");
  });

  it("does not double-encode title in /mind/evaluate-multi URL", async () => {
    const evaluator = new PaperEvaluatorMulti("http://localhost:8000");
    const params: EvaluationParams = {
      id: "1706.03762",
      title: "Attention Is All You Need",
      k_refs: 20,
      recommendations: 30,
      related: 5,
      llm_model: "gpt-4o-mini",
      filter_by_date: true,
      seed: 0,
    };

    const result = evaluator.startEvaluation(params);
    const url = MockEventSource.createdUrls[0];

    expect(url).toContain("/mind/evaluate-multi?");
    expect(url).toContain("title=Attention+Is+All+You+Need");
    expect(url).not.toContain("%2520");
    expect(url).not.toContain("%2B");

    evaluator.stopEvaluation();
    await expect(result).rejects.toThrow("Evaluation cancelled by user");
  });
});

describe("ArxivPaperService.searchPapers", () => {
  it("passes AbortSignal to fetch", async () => {
    const service = new ArxivPaperService("http://localhost:8000");
    const controller = new AbortController();

    const fetchMock = vi
      .fn<[RequestInfo | URL, RequestInit?], Promise<Response>>()
      .mockResolvedValue(
        new Response(
          JSON.stringify({
            query: "attention",
            total: 1,
            items: [
              {
                title: "Attention Is All You Need",
                arxiv_id: "1706.03762",
                abstract: "An abstract.",
                year: 2017,
                authors: ["A. Author"],
              },
            ],
          }),
          {
            status: 200,
            headers: { "Content-Type": "application/json" },
          },
        ),
      );

    const originalFetch = globalThis.fetch;
    globalThis.fetch = fetchMock as typeof fetch;

    try {
      await service.searchPapers("attention", 12, controller.signal);
      expect(fetchMock).toHaveBeenCalledOnce();
      const [, options] = fetchMock.mock.calls[0];
      expect(options?.signal).toBe(controller.signal);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("propagates AbortError from fetch", async () => {
    const service = new ArxivPaperService("http://localhost:8000");
    const controller = new AbortController();
    const abortError = Object.assign(new Error("The operation was aborted"), {
      name: "AbortError",
    });

    const fetchMock = vi
      .fn<[RequestInfo | URL, RequestInit?], Promise<Response>>()
      .mockRejectedValue(abortError);

    const originalFetch = globalThis.fetch;
    globalThis.fetch = fetchMock as typeof fetch;

    try {
      await expect(
        service.searchPapers("attention", 12, controller.signal),
      ).rejects.toMatchObject({
        name: "AbortError",
      });
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
});
