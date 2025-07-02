import { z } from "zod/v4";

// ============================================================================
// Schema Definitions
// ============================================================================

/**
 * Type of an entity in the hierarchical paper graph.
 */
export const EntityTypeSchema = z.enum([
  "title",
  "primary_area",
  "keyword",
  "tldr",
  "claim",
  "method",
  "experiment",
]);

/**
 * Text from the paper where the entity is mentioned.
 */
export const ExcerptSchema = z.object({
  /** Section (and nested subsections) where the text appears. */
  section: z.string(),
  /** Text that mentions the entity, copied verbatim. */
  text: z.string(),
});

/**
 * Entity in the hierarchical graph, representing a concept from the paper.
 */
export const EntitySchema = z.object({
  /** Optional detailed sentence about the entity. */
  detail: z.string().nullish(),
  /** Summary label of the entity. */
  label: z.string().min(1, "Label cannot be empty"),
  /** Type of the entity. */
  type: EntityTypeSchema,
  /** Optional extracted excerpts about the entity. */
  excerpts: z.array(ExcerptSchema).nullish(),
});

/**
 * Relationship between two nodes in the hierarchical graph.
 */
export const RelationshipSchema = z.object({
  /** Name of the source node. */
  source: z.string().min(1, "Source node name cannot be empty"),
  /** Name of the target node. */
  target: z.string().min(1, "Target node name cannot be empty"),
});

/**
 * Hierarchical graph extracted from a paper by an LLM.
 */
export const GraphSchema = z
  .object({
    /** Paper's abstract. */
    abstract: z.string(),
    /** Collection of entities (nodes) in the graph. */
    entities: z.array(EntitySchema).min(1, "Graph must have at least one entity"),
    /** Collection of relationships (edges) between entities. */
    relationships: z.array(RelationshipSchema),
    /** Paper's title. */
    title: z.string().min(1, "Title cannot be empty"),
    /**
     * String indicating the graph's validity. It's "Valid" if all rules pass,
     * "Empty graph" if there are no entities, or the first error message if invalid.
     */
    valid_status: z.string(),
    /**
     * Array of all validation error messages. Contains ["Valid"] if the graph is valid.
     */
    valid_status_all: z.array(z.string()).min(1),
  })
  .refine(
    data => {
      // Validate that all relationship nodes exist in entities
      const entityLabels = new Set(data.entities.map(e => e.label));
      return data.relationships.every(
        rel => entityLabels.has(rel.source) && entityLabels.has(rel.target),
      );
    },
    {
      message: "All relationship nodes must exist in the entities list",
    },
  );

/**
 * Section of a full paper, including its heading and text content.
 */
export const PaperSectionSchema = z.object({
  /** Heading of the section. */
  heading: z.string().min(1, "Section heading cannot be empty"),
  /** Full text content of the section. */
  text: z.string(),
});

/**
 * Source of evidence paper - matching backend PaperSource enum.
 */
export const EvidenceSourceSchema = z.enum(["citations", "semantic"]);

/**
 * Evidence item with paper citation information.
 */
export const EvidenceItemSchema = z.object({
  /** The evidence text or finding. */
  text: z.string(),
  /** ID of the paper this evidence comes from (e.g., S2 paper ID). */
  paper_id: z.string().nullable().optional(),
  /** Title of the paper this evidence comes from. */
  paper_title: z.string().nullable().optional(),
  /** Source of the related paper (citations or semantic similarity). */
  source: EvidenceSourceSchema.nullable().optional(),
});

/**
 * Structured evaluation of a paper's novelty with supporting and contradictory evidence.
 */
export const StructuredEvalSchema = z.object({
  /** Brief summary of the paper's main contributions and approach */
  paper_summary: z.string(),
  /** List of evidence from related papers that support the paper's novelty */
  supporting_evidence: z.array(z.union([z.string(), EvidenceItemSchema])),
  /** List of evidence from related papers that contradict the paper's novelty */
  contradictory_evidence: z.array(z.union([z.string(), EvidenceItemSchema])),
  /** Key technical comparisons that influenced the novelty decision */
  key_comparisons: z.array(z.string()).optional().default([]),
  /** Final assessment of the paper's novelty based on the evidence */
  conclusion: z.string(),
  /** 1 if the paper is novel, or 0 if it's not novel */
  label: z.int(),
  /** Probability that the paper is novel. */
  probability: z.number().nullish(),
});

/**
 * Paper with its metadata and evaluation results.
 */
export const PaperSchema = z.object({
  /** Abstract of the paper. */
  abstract: z.string(),
  /** Final approval decision for the paper (approved/rejected). Optional. */
  approval: z.boolean().nullish(),
  /** Names of the paper's authors. */
  authors: z.array(z.string()).min(1, "Paper must have at least one author"),
  /** Conference where the paper was submitted (optional). */
  conference: z.string().nullish(),
  /** Unique identifier for the paper. */
  id: z.string(),
  /** Novelty rating (1-5) from the main human review. */
  rating: z.int("Rating must be an integer"),
  /** Rationale generated by the model for its novelty prediction. */
  rationale_pred: z.string(),
  /** Original rationale provided by a human reviewer. */
  rationale_true: z.string(),
  /** Sections of the paper's main text. */
  sections: z.array(PaperSectionSchema),
  /** Title of the paper. */
  title: z.string().min(1, "Title cannot be empty"),
  /** Model's predicted novelty label. */
  y_pred: z.int("y_pred must be an integer"),
  /** Ground truth novelty label from human annotation. */
  y_true: z.int("y_true must be an integer"),
  /** Publication year of the paper. */
  year: z.int("Year must be an integer").min(1900, "Year must be after 1900"),
  /** Rationale structured in parts (optional). */
  structured_evaluation: StructuredEvalSchema.nullish(),
  /** Paper arXiv ID (optional) */
  arxiv_id: z.string().nullish(),
});

/**
 * Polarity of a citation context, indicating a positive or negative relationship.
 */
export const ContextPolaritySchema = z.enum(["positive", "negative"]);

/**
 * Source of a related paper, either from citations or semantic similarity.
 */
export const RelatedPaperSourceSchema = z.enum(["semantic", "citations"]);

/**
 * Citation context with sentence and polarity information.
 */
export const CitationContextSchema = z.object({
  sentence: z.string(),
  polarity: ContextPolaritySchema.nullable(),
});

/**
 * Related paper with its summary, used as context for evaluation.
 */
export const RelatedPaperSchema = z.object({
  /** Abstract of the related paper. */
  abstract: z.string(),
  /** Unique identifier of the related paper. */
  paper_id: z.string().regex(/^[a-zA-Z0-9_-]+$/, "Invalid paper ID format"),
  /** Polarity of the relationship (positive/negative) to the main paper. */
  polarity: ContextPolaritySchema,
  /** Similarity score between the main and related paper. */
  score: z.number().transform(val => Math.max(0, Math.min(1, val))),
  /** Source from which this related paper was retrieved. */
  source: RelatedPaperSourceSchema,
  /** Summary of the related paper's key points relevant to the main paper. */
  summary: z.string().min(1, "Summary cannot be empty"),
  /** Title of the related paper. */
  title: z.string().min(1, "Title cannot be empty"),
  /** Publication year of the related paper. */
  year: z.number().nullish(),
  /** Authors of the related paper. */
  authors: z.array(z.string()).nullish(),
  /** Publication venue of the related paper. */
  venue: z.string().nullish(),
  /** Number of papers that cite this related paper. */
  citation_count: z.number().nullish(),
  /** Number of papers this related paper references. */
  reference_count: z.number().nullish(),
  /** Number of influential citations for this related paper. */
  influential_citation_count: z.number().nullish(),
  /** Semantic Scholar's secondary identifier for this related paper. */
  corpus_id: z.number().nullish(),
  /** URL to the related paper on Semantic Scholar website. */
  url: z.string().nullish(),
  /** arXiv identifier for this related paper if available. */
  arxiv_id: z.string().nullish(),
  /** Citation contexts for citation-based papers. */
  contexts: z.array(CitationContextSchema).nullish(),
  /** Background text for semantic papers (background matching). */
  background: z.string().nullish(),
  /** Target text for semantic papers (target matching). */
  target: z.string().nullish(),
});

/**
 * Represents a directed 'used for' relation between two scientific terms.
 * Relations are always (head, used-for, tail).
 */
export const PaperTermRelationSchema = z.object({
  /** Head term of the relation. */
  head: z.string().min(1, "Head term cannot be empty"),
  /** Tail term of the relation. */
  tail: z.string().min(1, "Tail term cannot be empty"),
});

/**
 * Structured output for scientific term extraction.
 */
export const PaperTermsSchema = z.object({
  /** Core problems, objectives or applications addressed. */
  tasks: z.array(z.string()),
  /** Technical approaches, algorithms, or frameworks used/proposed. */
  methods: z.array(z.string()),
  /** Evaluation metrics and measures mentioned. */
  metrics: z.array(z.string()),
  /** Datasets, resources, or tools utilised. */
  resources: z.array(z.string()),
  /** Directed relations between terms. */
  relations: z.array(PaperTermRelationSchema),
});

/**
 * Container for all data needed for a demonstration, combining the main paper's
 * extracted graph, its evaluation results, and its summarised related papers.
 */
export const GraphResultSchema = z.object({
  /** The hierarchical graph extracted from the main paper. */
  graph: GraphSchema,
  /** The main paper with its metadata and evaluation results. */
  paper: PaperSchema,
  /** A list of related papers with their summaries. */
  related: z.array(RelatedPaperSchema),
  /** Structured scientific terms extracted from the paper (optional). */
  terms: PaperTermsSchema.nullish(),
  /** Background information about the paper (optional). */
  background: z.string().nullish(),
  /** Target information about the paper (optional). */
  target: z.string().nullish(),
});

/**
 * Item in paper search results.
 */
export const PaperSearchItemSchema = z.object({
  /** Title of the paper on arXiv. */
  title: z.string(),
  /** arXiv paper ID. */
  arxiv_id: z.string(),
  /** Abstract on arXiv. */
  abstract: z.string(),
  /** Year of submission to arXiv */
  year: z.int().nullable(),
  /** Paper authors */
  authors: z.array(z.string()),
});

/*
 * Results from the search query.
 */
export const PaperSearchResultsSchema = z.object({
  /** Search results. */
  items: z.array(PaperSearchItemSchema),
  /** Query used for the search. */
  query: z.string(),
  /** How many items were retrieved. */
  total: z.int(),
});

/**
 * Results from full paper evaluation.
 */
export const EvalResultSchema = z.object({
  /** Evaluate paper graph. */
  result: GraphResultSchema,
  /** Total cost of evaluation, including annotation. */
  cost: z.number(),
});

/**
 * Parameters for paper evaluation request
 */
export const EvaluationParamsSchema = z.object({
  /** arXiv ID of the paper to analyse */
  id: z.string(),
  /** Title of the paper on arXiv */
  title: z.string(),
  /** Number of references to analyse */
  k_refs: z.number().min(10).max(50).optional().default(20),
  /** Number of recommended papers to retrieve */
  recommendations: z.number().min(5).max(50).optional().default(30),
  /** Number of related papers per type */
  related: z.number().min(1).max(10).optional().default(3),
  /** LLM model to use */
  llm_model: z
    .enum(["gpt-4o", "gpt-4o-mini", "gemini-2.0-flash"])
    .optional()
    .default("gpt-4o-mini"),
  /** Filter recommended papers to only include those published before the main paper */
  filter_by_date: z.boolean().optional().default(true),
  /** Random seed for reproducible results */
  seed: z.number().optional().default(0),
});

/**
 * Server-Sent Event data structure for evaluation progress
 */
export const SSEEventDataSchema = z.object({
  /** Optional message for progress/error events */
  message: z.string().optional(),
  /** Result data for complete events (EvalResult already contains cost field) */
  result: EvalResultSchema.optional(),
});

/**
 * Response from partial paper evaluation.
 */
export const PartialEvaluationResponseSchema = z.object({
  /** Paper title. */
  title: z.string(),
  /** Paper abstract. */
  abstract: z.string(),
  /** Binary novelty score (1=novel, 0=not novel). */
  label: z.int(),
  /** Percentage chance of the paper being novel. */
  probability: z.number().nullish(),
  /** Summary of paper contributions. */
  paper_summary: z.string(),
  /** Evidence supporting novelty. */
  supporting_evidence: z.array(EvidenceItemSchema),
  /** Evidence contradicting novelty. */
  contradictory_evidence: z.array(EvidenceItemSchema),
  /** Final assessment. */
  conclusion: z.string(),
  /** Total GPT cost. */
  total_cost: z.number(),
  /** Related papers. */
  related: z.array(RelatedPaperSchema),
});

// ============================================================================
// Type Exports (Inferred from Schemas)
// ============================================================================

export type Excerpt = z.infer<typeof ExcerptSchema>;
export type EntityType = z.infer<typeof EntityTypeSchema>;
export type Entity = z.infer<typeof EntitySchema>;
export type Relationship = z.infer<typeof RelationshipSchema>;
export type Graph = z.infer<typeof GraphSchema>;
export type PaperSection = z.infer<typeof PaperSectionSchema>;
export type EvidenceSource = z.infer<typeof EvidenceSourceSchema>;
export type EvidenceItem = z.infer<typeof EvidenceItemSchema>;
export type StructuredEval = z.infer<typeof StructuredEvalSchema>;
export type Paper = z.infer<typeof PaperSchema>;
export type ContextPolarity = z.infer<typeof ContextPolaritySchema>;
export type RelatedPaperSource = z.infer<typeof RelatedPaperSourceSchema>;
export type CitationContext = z.infer<typeof CitationContextSchema>;
export type RelatedPaper = z.infer<typeof RelatedPaperSchema>;
export type PaperTermRelation = z.infer<typeof PaperTermRelationSchema>;
export type PaperTerms = z.infer<typeof PaperTermsSchema>;
export type GraphResult = z.infer<typeof GraphResultSchema>;
export type PaperSearchItem = z.infer<typeof PaperSearchItemSchema>;
export type PaperSearchResults = z.infer<typeof PaperSearchResultsSchema>;
export type EvalResult = z.infer<typeof EvalResultSchema>;
export type EvaluationParams = z.infer<typeof EvaluationParamsSchema>;
export type SSEEventData = z.infer<typeof SSEEventDataSchema>;
export type PartialEvaluationResponse = z.infer<typeof PartialEvaluationResponseSchema>;
