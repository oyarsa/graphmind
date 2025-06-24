import { z } from "zod";

export const PaperSchema = z.object({
  id: z.string(),
  title: z.string(),
  authors: z.array(z.string()),
  year: z.number(),
  abstract: z.string(),
  venue: z.string(),
  citation_count: z.number(),
  doi: z.string().optional(),
  pdf_url: z.string().optional(),
});

export type Paper = z.infer<typeof PaperSchema>;

export const SearchResultSchema = z.object({
  query: z.string(),
  results: z.array(
    z.object({
      id: z.string(),
      title: z.string(),
      year: z.number(),
      authors: z.array(z.string()),
      relevance: z.number(),
    }),
  ),
  total: z.number(),
});

export type SearchResult = z.infer<typeof SearchResultSchema>;

export const PaperNeighbourSchema = PaperSchema.extend({
  similarity: z.number(),
});

export type PaperNeighbour = z.infer<typeof PaperNeighbourSchema>;

export const RelatedPapersResponseSchema = z.object({
  paper_id: z.string(),
  neighbours: z.array(PaperNeighbourSchema),
  total_papers: z.number(),
});

export type RelatedPapersResponse = z.infer<typeof RelatedPapersResponseSchema>;

export type LinkType = "cited" | "similar";

export interface GraphLink {
  source: string | Paper;
  target: string | Paper;
  type: LinkType;
  similarity: number;
}

export type D3Node = Paper & d3.SimulationNodeDatum;

export interface D3Link extends d3.SimulationLinkDatum<D3Node> {
  type: LinkType;
  similarity: number;
}

export function getNodeId(node: string | number | D3Node): string {
  return typeof node === "string" || typeof node === "number"
    ? node.toString()
    : node.id;
}
export function getNodePos(node: string | number | D3Node): { x?: number; y?: number } {
  return typeof node === "string" || typeof node === "number"
    ? {}
    : { x: node.x, y: node.y };
}

export abstract class DataService {
  abstract searchPapers(query: string, limit?: number): Promise<SearchResult>;

  abstract getPaperDetails(id: string): Promise<Paper | null>;

  abstract getCitedPapers(id: string, limit?: number): Promise<RelatedPapersResponse>;

  abstract getSemanticPapers(
    id: string,
    limit?: number,
  ): Promise<RelatedPapersResponse>;
}
