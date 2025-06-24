import { describe, it, expect } from "vitest";
import {
  createPaperTermsDisplay,
  getRelationshipStyle,
  getScoreDisplay,
  formatTypeName,
} from "./helpers";
import { RelatedPaper, PaperTerms } from "./model";

describe("createPaperTermsDisplay", () => {
  it("should return no data message when all inputs are null/empty", () => {
    const result = createPaperTermsDisplay(null, null, null, null);
    expect(result).toContain("No analysis data available");
  });

  it("should display primary area when provided", () => {
    const result = createPaperTermsDisplay(null, null, null, "Machine Learning");
    expect(result).toContain("Primary Area");
    expect(result).toContain("Machine Learning");
    expect(result).toContain("bg-green-500");
  });

  it("should display background when provided", () => {
    const result = createPaperTermsDisplay(null, "Background text", null, null);
    expect(result).toContain("Background");
    expect(result).toContain("Background text");
    expect(result).toContain("bg-blue-500");
  });

  it("should display target when provided", () => {
    const result = createPaperTermsDisplay(null, null, "Target text", null);
    expect(result).toContain("Target");
    expect(result).toContain("Target text");
    expect(result).toContain("bg-purple-500");
  });

  it("should display terms sections when provided", () => {
    const terms: PaperTerms = {
      tasks: ["classification", "detection"],
      methods: ["CNN", "transformer"],
      metrics: ["accuracy", "F1-score"],
      resources: ["dataset1", "dataset2"],
      relations: [],
    };

    const result = createPaperTermsDisplay(terms, null, null, null);

    expect(result).toContain("Tasks");
    expect(result).toContain("classification");
    expect(result).toContain("detection");

    expect(result).toContain("Methods");
    expect(result).toContain("CNN");
    expect(result).toContain("transformer");

    expect(result).toContain("Metrics");
    expect(result).toContain("accuracy");
    expect(result).toContain("F1-score");

    expect(result).toContain("Resources");
    expect(result).toContain("dataset1");
    expect(result).toContain("dataset2");
  });

  it("should skip empty term sections", () => {
    const terms: PaperTerms = {
      tasks: ["classification"],
      methods: [],
      metrics: [],
      resources: [],
      relations: [],
    };

    const result = createPaperTermsDisplay(terms, null, null, null);

    expect(result).toContain("Tasks");
    expect(result).not.toContain("🔧 Methods");
    expect(result).not.toContain("📊 Metrics");
    expect(result).not.toContain("📚 Resources");
  });

  it("should display both background sections and terms", () => {
    const terms: PaperTerms = {
      tasks: ["classification"],
      methods: [],
      metrics: [],
      resources: [],
      relations: [],
    };

    const result = createPaperTermsDisplay(terms, "Background", "Target", "ML");

    expect(result).toContain("Primary Area");
    expect(result).toContain("Background");
    expect(result).toContain("Target");
    expect(result).toContain("Tasks");
    expect(result).toContain("mt-4"); // Should add spacing when both sections present
  });
});

describe("getRelationshipStyle", () => {
  it("should return target style for semantic positive", () => {
    const paper: RelatedPaper = {
      paper_id: "1",
      title: "Test",
      abstract: "Test",
      summary: "Test",
      score: 0.8,
      source: "semantic",
      polarity: "positive",
    };

    const result = getRelationshipStyle(paper);

    expect(result.type).toBe("target");
    expect(result.label).toBe("Target");
    expect(result.icon).toBe("🧠");
    expect(result.style).toBe("rounded-full");
    expect(result.color).toContain("bg-orange-100");
  });

  it("should return background style for semantic negative", () => {
    const paper: RelatedPaper = {
      paper_id: "1",
      title: "Test",
      abstract: "Test",
      summary: "Test",
      score: 0.8,
      source: "semantic",
      polarity: "negative",
    };

    const result = getRelationshipStyle(paper);

    expect(result.type).toBe("background");
    expect(result.label).toBe("Background");
    expect(result.icon).toBe("🧠");
    expect(result.style).toBe("rounded-full");
    expect(result.color).toContain("bg-green-100");
  });

  it("should return supporting style for citations positive", () => {
    const paper: RelatedPaper = {
      paper_id: "1",
      title: "Test",
      abstract: "Test",
      summary: "Test",
      score: 0.8,
      source: "citations",
      polarity: "positive",
    };

    const result = getRelationshipStyle(paper);

    expect(result.type).toBe("supporting");
    expect(result.label).toBe("Supporting");
    expect(result.icon).toBe("🔗");
    expect(result.style).toBe("rounded-md");
    expect(result.color).toContain("bg-emerald-100");
  });

  it("should return contrasting style for citations negative", () => {
    const paper: RelatedPaper = {
      paper_id: "1",
      title: "Test",
      abstract: "Test",
      summary: "Test",
      score: 0.8,
      source: "citations",
      polarity: "negative",
    };

    const result = getRelationshipStyle(paper);

    expect(result.type).toBe("contrasting");
    expect(result.label).toBe("Contrasting");
    expect(result.icon).toBe("🔗");
    expect(result.style).toBe("rounded-md");
    expect(result.color).toContain("bg-red-100");
  });

  it("should default to contrasting for unknown combinations", () => {
    const paper: RelatedPaper = {
      paper_id: "1",
      title: "Test",
      abstract: "Test",
      summary: "Test",
      score: 0.8,
      source: "unknown" as "semantic" | "citations",
      polarity: "unknown" as "positive" | "negative",
    };

    const result = getRelationshipStyle(paper);

    expect(result.type).toBe("contrasting");
    expect(result.label).toBe("Contrasting");
  });
});

describe("getScoreDisplay", () => {
  it("should return green for high scores", () => {
    const result = getScoreDisplay(0.8);

    expect(result.scorePercent).toBe(80);
    expect(result.scoreColor).toBe("bg-green-500");
  });

  it("should return yellow for medium scores", () => {
    const result = getScoreDisplay(0.5);

    expect(result.scorePercent).toBe(50);
    expect(result.scoreColor).toBe("bg-yellow-500");
  });

  it("should return red for low scores", () => {
    const result = getScoreDisplay(0.2);

    expect(result.scorePercent).toBe(20);
    expect(result.scoreColor).toBe("bg-red-500");
  });

  it("should handle edge cases", () => {
    expect(getScoreDisplay(0.7).scoreColor).toBe("bg-green-500");
    expect(getScoreDisplay(0.4).scoreColor).toBe("bg-yellow-500");
    expect(getScoreDisplay(0.0).scoreColor).toBe("bg-red-500");
    expect(getScoreDisplay(1.0).scoreColor).toBe("bg-green-500");
  });

  it("should round percentages correctly", () => {
    expect(getScoreDisplay(0.456).scorePercent).toBe(46);
    expect(getScoreDisplay(0.454).scorePercent).toBe(45);
    expect(getScoreDisplay(0.999).scorePercent).toBe(100);
  });
});

describe("formatTypeName", () => {
  it("should capitalize and replace underscores", () => {
    expect(formatTypeName("primary_area")).toBe("Primary area");
    expect(formatTypeName("method")).toBe("Method");
    expect(formatTypeName("experiment")).toBe("Experiment");
  });

  it("should handle single words", () => {
    expect(formatTypeName("title")).toBe("Title");
    expect(formatTypeName("claim")).toBe("Claim");
  });

  it("should handle multiple underscores", () => {
    expect(formatTypeName("some_long_type_name")).toBe("Some long type name");
  });

  it("should handle empty string", () => {
    expect(formatTypeName("")).toBe("");
  });

  it("should handle already formatted strings", () => {
    expect(formatTypeName("AlreadyFormatted")).toBe("AlreadyFormatted");
  });
});
