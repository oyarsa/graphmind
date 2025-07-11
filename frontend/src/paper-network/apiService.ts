import {
  DataService,
  type Paper,
  type SearchResult,
  type RelatedPapersResponse,
  PaperSchema,
  SearchResultSchema,
  RelatedPapersResponseSchema,
} from "./model";
import { assertResponse } from "../util";

export class ApiService extends DataService {
  constructor(private baseUrl: string) {
    super();
  }

  async searchPapers(query: string, limit = 20): Promise<SearchResult> {
    const encodedQuery = encodeURIComponent(query);
    const response = await fetch(
      `${this.baseUrl}/network/search?q=${encodedQuery}&limit=${limit}`,
    );
    assertResponse(response, "Search failed");
    return SearchResultSchema.parse(await response.json());
  }

  async getPaperDetails(id: string): Promise<Paper | null> {
    const response = await fetch(`${this.baseUrl}/network/papers/${id}`);
    if (response.status === 404) {
      return null;
    }
    assertResponse(response, "Failed to get paper details");
    return PaperSchema.parse(await response.json());
  }

  async getCitedPapers(id: string, limit = 20): Promise<RelatedPapersResponse> {
    const response = await fetch(
      `${this.baseUrl}/network/related/${id}?type=citation&limit=${limit}`,
    );
    assertResponse(response, "Failed to get cited papers");
    return RelatedPapersResponseSchema.parse(await response.json());
  }

  async getSemanticPapers(id: string, limit = 20): Promise<RelatedPapersResponse> {
    const response = await fetch(
      `${this.baseUrl}/network/related/${id}?type=semantic&limit=${limit}`,
    );
    assertResponse(response, "Failed to get semantic papers");
    return RelatedPapersResponseSchema.parse(await response.json());
  }
}
