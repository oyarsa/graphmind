import { describe, it, expect } from "vitest";
import {
  createPaperTermsDisplay,
  getRelationshipStyle,
  getScoreDisplay,
  formatTypeName,
  formatPaperCitation,
} from "./helpers";
import { RelatedPaper } from "./model";

// Helper function to create test related papers with all required fields
function createTestRelatedPaper(overrides: Partial<RelatedPaper>): RelatedPaper {
  return {
    paper_id: "test-id",
    title: "Test Paper",
    abstract: "Test abstract",
    summary: "Test summary",
    score: 0.8,
    source: "semantic",
    polarity: "positive",
    ...overrides,
  };
}

describe("createPaperTermsDisplay", () => {
  it("should return no data message when all inputs are null/empty", () => {
    const result = createPaperTermsDisplay(null, null, null);
    expect(result).toContain("No analysis data available");
  });

  it("should display primary area when provided", () => {
    const result = createPaperTermsDisplay(null, null, "Machine Learning");
    expect(result).toContain("Primary Area");
    expect(result).toContain("Machine Learning");
    expect(result).toContain("bg-green-500");
  });

  it("should display background when provided", () => {
    const result = createPaperTermsDisplay("Background text", null, null);
    expect(result).toContain("Background");
    expect(result).toContain("Background text");
    expect(result).toContain("bg-blue-500");
  });

  it("should display target when provided", () => {
    const result = createPaperTermsDisplay(null, "Target text", null);
    expect(result).toContain("Target");
    expect(result).toContain("Target text");
    expect(result).toContain("bg-purple-500");
  });

  it("should only display background sections when provided", () => {
    const result = createPaperTermsDisplay("Background", "Target", "ML");

    expect(result).toContain("Primary Area");
    expect(result).toContain("Background");
    expect(result).toContain("Target");
  });
});

describe("getRelationshipStyle", () => {
  it("should return target style for semantic positive", () => {
    const paper = createTestRelatedPaper({
      paper_id: "1",
      title: "Test",
      abstract: "Test",
      summary: "Test",
      score: 0.8,
      source: "semantic",
      polarity: "positive",
    });

    const result = getRelationshipStyle(paper);

    expect(result.type).toBe("target");
    expect(result.label).toBe("Target");
    expect(result.icon).toBe("🧠");
    expect(result.style).toBe("rounded-full");
    expect(result.color).toContain("bg-orange-100");
  });

  it("should return background style for semantic negative", () => {
    const paper = createTestRelatedPaper({
      paper_id: "1",
      title: "Test",
      abstract: "Test",
      summary: "Test",
      score: 0.8,
      source: "semantic",
      polarity: "negative",
    });

    const result = getRelationshipStyle(paper);

    expect(result.type).toBe("background");
    expect(result.label).toBe("Background");
    expect(result.icon).toBe("🧠");
    expect(result.style).toBe("rounded-full");
    expect(result.color).toContain("bg-green-100");
  });

  it("should return supporting style for citations positive", () => {
    const paper = createTestRelatedPaper({
      paper_id: "1",
      title: "Test",
      abstract: "Test",
      summary: "Test",
      score: 0.8,
      source: "citations",
      polarity: "positive",
    });

    const result = getRelationshipStyle(paper);

    expect(result.type).toBe("supporting");
    expect(result.label).toBe("Supporting");
    expect(result.icon).toBe("🔗");
    expect(result.style).toBe("rounded-md");
    expect(result.color).toContain("bg-emerald-100");
  });

  it("should return contrasting style for citations negative", () => {
    const paper = createTestRelatedPaper({
      paper_id: "1",
      title: "Test",
      abstract: "Test",
      summary: "Test",
      score: 0.8,
      source: "citations",
      polarity: "negative",
    });

    const result = getRelationshipStyle(paper);

    expect(result.type).toBe("contrasting");
    expect(result.label).toBe("Contrasting");
    expect(result.icon).toBe("🔗");
    expect(result.style).toBe("rounded-md");
    expect(result.color).toContain("bg-red-100");
  });

  it("should default to contrasting for unknown combinations", () => {
    const paper = createTestRelatedPaper({
      paper_id: "1",
      title: "Test",
      abstract: "Test",
      summary: "Test",
      score: 0.8,
      source: "unknown" as "semantic" | "citations",
      polarity: "unknown" as "positive" | "negative",
    });

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

describe("formatPaperCitation", () => {
  it("should format with single author and year", () => {
    const paper = createTestRelatedPaper({
      authors: ["Smith"],
      year: 2023,
      title: "A Very Long Paper Title That Should Be Truncated",
    });

    const result = formatPaperCitation(paper);
    expect(result).toBe("Smith (2023)");
  });

  it("should format with multiple authors and year", () => {
    const paper = createTestRelatedPaper({
      authors: ["Smith", "Johnson", "Williams"],
      year: 2022,
      title: "A Very Long Paper Title That Should Be Truncated",
    });

    const result = formatPaperCitation(paper);
    expect(result).toBe("Smith et al. (2022)");
  });

  it("should fallback to truncated title when no authors", () => {
    const paper = createTestRelatedPaper({
      authors: null,
      year: 2023,
      title: "A Very Long Paper Title That Should Be Truncated",
    });

    const result = formatPaperCitation(paper);
    expect(result).toBe("A Very Long Paper Ti...");
  });

  it("should fallback to truncated title when no year", () => {
    const paper = createTestRelatedPaper({
      authors: ["Smith"],
      year: null,
      title: "A Very Long Paper Title That Should Be Truncated",
    });

    const result = formatPaperCitation(paper);
    expect(result).toBe("A Very Long Paper Ti...");
  });

  it("should fallback to truncated title when empty authors array", () => {
    const paper = createTestRelatedPaper({
      authors: [],
      year: 2023,
      title: "A Very Long Paper Title That Should Be Truncated",
    });

    const result = formatPaperCitation(paper);
    expect(result).toBe("A Very Long Paper Ti...");
  });

  it("should return full title when shorter than 20 characters", () => {
    const paper = createTestRelatedPaper({
      authors: null,
      year: null,
      title: "Short Title",
    });

    const result = formatPaperCitation(paper);
    expect(result).toBe("Short Title");
  });

  it("should handle exactly 20 character title", () => {
    const paper = createTestRelatedPaper({
      authors: null,
      year: null,
      title: "This is exactly 20c", // Exactly 20 characters
    });

    const result = formatPaperCitation(paper);
    expect(result).toBe("This is exactly 20c");
  });
});
