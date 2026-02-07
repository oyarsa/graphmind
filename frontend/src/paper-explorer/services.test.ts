import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { type EvaluationParams } from "./model";
import { PaperEvaluator, PaperEvaluatorMulti } from "./services";

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
