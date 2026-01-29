import { describe, it, expect } from "vitest";
import {
  createPaperTermsDisplay,
  getRelationshipStyle,
  getScoreDisplay,
  formatTypeName,
  formatPaperCitation,
  stripLatexCitations,
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

describe("stripLatexCitations", () => {
  it("should return empty string for empty input", () => {
    expect(stripLatexCitations("")).toBe("");
  });

  it("should remove ~\\citep{} commands", () => {
    const input = "This is some text~\\citep{ref2020} with a citation.";
    expect(stripLatexCitations(input)).toBe("This is some text with a citation.");
  });

  it("should remove \\citep{} commands without tilde", () => {
    const input = "This is some text\\citep{ref2020} with a citation.";
    expect(stripLatexCitations(input)).toBe("This is some text with a citation.");
  });

  it("should remove ~\\cite{} commands", () => {
    const input = "As shown in~\\cite{smith2021}, the results are clear.";
    expect(stripLatexCitations(input)).toBe("As shown in, the results are clear.");
  });

  it("should remove \\citet{} commands", () => {
    const input = "According to \\citet{jones2022}, this works.";
    expect(stripLatexCitations(input)).toBe("According to, this works.");
  });

  it("should handle multiple citations", () => {
    const input = "Some text~\\citep{ref1} and more~\\cite{ref2} with \\citet{ref3}.";
    expect(stripLatexCitations(input)).toBe("Some text and more with.");
  });

  it("should handle citations with multiple references", () => {
    const input = "This is cited~\\citep{DBLP:conf/emnlp/QiZWZYLHLB23,DBLP:ref2}.";
    expect(stripLatexCitations(input)).toBe("This is cited.");
  });

  it("should preserve text without citations", () => {
    const input = "This is plain text without any citations.";
    expect(stripLatexCitations(input)).toBe(
      "This is plain text without any citations.",
    );
  });

  it("should clean up double spaces after removal", () => {
    const input = "Text  ~\\citep{ref}  with  spaces.";
    expect(stripLatexCitations(input)).toBe("Text with spaces.");
  });

  // Tests for malformed LaTeX (space after backslash)
  it("should remove malformed citations with space after backslash", () => {
    const input = "compared to the baseline, PAIE~\\ cite{PAIE}.";
    expect(stripLatexCitations(input)).toBe("compared to the baseline, PAIE.");
  });

  it("should remove malformed citep with space after backslash", () => {
    const input = "as shown~\\ citep{ref2020} in the paper.";
    expect(stripLatexCitations(input)).toBe("as shown in the paper.");
  });

  // Tests for footnotes
  it("should remove \\footnote{} commands", () => {
    const input = "This is important\\footnote{See appendix for details}.";
    expect(stripLatexCitations(input)).toBe("This is important.");
  });

  it("should remove ~\\footnote{} commands with tilde", () => {
    const input = "SLMs~\\footnote{All code is available at https://github.com}.";
    expect(stripLatexCitations(input)).toBe("SLMs.");
  });

  it("should remove malformed footnotes with space after backslash", () => {
    const input = "performance~\\ footnote{All code is available}.";
    expect(stripLatexCitations(input)).toBe("performance.");
  });

  // Tests for other LaTeX commands
  it("should remove \\textbf{} commands", () => {
    const input = "This is \\textbf{bold} text.";
    expect(stripLatexCitations(input)).toBe("This is text.");
  });

  it("should remove \\emph{} commands", () => {
    const input = "This is \\emph{emphasised} text.";
    expect(stripLatexCitations(input)).toBe("This is text.");
  });

  // Tests for standalone tildes
  it("should replace standalone tildes with spaces", () => {
    const input = "Figure~1 shows the results.";
    expect(stripLatexCitations(input)).toBe("Figure 1 shows the results.");
  });

  // Real-world examples from the user
  it("should handle real example with malformed cite and footnote", () => {
    const input =
      "The CsEAE model achieved improvements of 2.1% compared to PAIE~\\ cite{PAIE}. " +
      "For LLMs, performance is comparable to SLMs~\\ footnote{Code at github.com}.";
    expect(stripLatexCitations(input)).toBe(
      "The CsEAE model achieved improvements of 2.1% compared to PAIE. " +
        "For LLMs, performance is comparable to SLMs.",
    );
  });

  it("should handle real citation context example", () => {
    const input =
      "Reinforcement Learning through Human Feedback (PPO) has seen " +
      "applications for instruction tuning~\\citep{Castricato2022,Shulev2024}";
    expect(stripLatexCitations(input)).toBe(
      "Reinforcement Learning through Human Feedback (PPO) has seen " +
        "applications for instruction tuning",
    );
  });
});
