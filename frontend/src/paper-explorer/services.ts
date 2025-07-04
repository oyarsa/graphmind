import { z } from "zod/v4";
import {
  GraphResult,
  GraphResultSchema,
  PaperSearchResults,
  PaperSearchResultsSchema,
  EvalResult,
  EvalResultSchema,
  EvaluationParams,
  EvaluationParamsSchema,
  SSEEventDataSchema,
  PartialSSEEventDataSchema,
  PartialEvaluationResponse,
} from "./model";

/**
 * Service for managing the static JSON paper dataset.
 * Handles loading and local operations on the dataset.
 *
 * Uses Vite's BASE_URL to properly resolve paths in both development and production:
 * - Development: BASE_URL = "/" → "data/test.json" → "/data/test.json"
 * - Production: BASE_URL = "/graphmind/" → "data/test.json" → "/graphmind/data/test.json"
 */
export class JsonPaperDataset {
  constructor(private jsonPath: string) {}

  async loadDataset(): Promise<GraphResult[]> {
    // Combine base URL with the relative path for proper resolution
    const fullPath = import.meta.env.BASE_URL + this.jsonPath;

    const response = await fetch(fullPath);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    let jsonData: unknown;

    try {
      // Try parsing as JSON first (for local/decompressed case)
      jsonData = await response.clone().json();
    } catch {
      // If that fails, try decompressing (for GitHub Pages case)
      const compressedData = await response.arrayBuffer();
      const decompressedStream = new Response(
        new Blob([compressedData])
          .stream()
          .pipeThrough(new DecompressionStream("gzip")),
      );
      jsonData = await decompressedStream.json();
    }

    const GraphResultArraySchema = z.array(GraphResultSchema);
    return GraphResultArraySchema.parse(jsonData);
  }
}

/**
 * Service for arXiv paper discovery and fetching.
 * Handles API communication with the arXiv search backend.
 */
export class ArxivPaperService {
  constructor(private baseUrl: string) {}

  async searchPapers(query: string, limit = 12): Promise<PaperSearchResults> {
    const encodedQuery = encodeURIComponent(query);
    const response = await fetch(
      `${this.baseUrl}/mind/search?q=${encodedQuery}&limit=${limit}`,
    );

    if (!response.ok) {
      if (response.status === 503) {
        throw new Error("Error from arXiv. Try again later.");
      }
      throw new Error(`ArXiv search failed: ${response.status} ${response.statusText}`);
    }

    return PaperSearchResultsSchema.parse(await response.json());
  }

  /**
   * Evaluate a paper to get full information including graph and analysis.
   * This operation can take 1-2 minutes to complete.
   * @deprecated Use PaperEvaluator for real-time progress updates
   */
  async evaluatePaper(arxivId: string, title: string): Promise<EvalResult> {
    const encodedId = encodeURIComponent(arxivId);
    const encodedTitle = encodeURIComponent(title);
    const response = await fetch(
      `${this.baseUrl}/mind/evaluate?id=${encodedId}&title=${encodedTitle}`,
    );

    if (!response.ok) {
      throw new Error(
        `Paper evaluation failed: ${response.status} ${response.statusText}`,
      );
    }

    return EvalResultSchema.parse(await response.json());
  }
}

/**
 * Real-time paper evaluator using Server-Sent Events.
 * Provides live progress updates during paper evaluation.
 */
export class PaperEvaluator {
  private eventSource: EventSource | null = null;
  private isEvaluating = false;
  private currentReject?: (reason?: unknown) => void;

  onConnected?: (message: string) => void;
  onProgress?: (message: string, progress: number) => void;
  onComplete?: (result: EvalResult) => void;
  onError?: (error: string) => void;
  onConnectionError?: (event: Event) => void;

  constructor(private baseUrl: string) {}

  /**
   * Start paper evaluation with real-time progress updates
   */
  async startEvaluation(paperData: EvaluationParams): Promise<EvalResult> {
    if (this.isEvaluating) {
      throw new Error("Evaluation already in progress");
    }

    // Validate input parameters
    const validatedParams = EvaluationParamsSchema.parse(paperData);

    const encodedId = encodeURIComponent(validatedParams.id);
    const encodedTitle = encodeURIComponent(validatedParams.title);
    const params = new URLSearchParams({
      id: encodedId,
      title: encodedTitle,
      k_refs: validatedParams.k_refs.toString(),
      recommendations: validatedParams.recommendations.toString(),
      related: validatedParams.related.toString(),
      llm_model: validatedParams.llm_model,
      seed: validatedParams.seed.toString(),
    });

    const url = `${this.baseUrl}/mind/evaluate?${params}`;
    this.eventSource = new EventSource(url);
    this.isEvaluating = true;

    return new Promise((resolve, reject) => {
      // Store reject function for cancellation
      this.currentReject = reject;

      const eventSource = this.eventSource;
      if (!eventSource) {
        reject(new Error("EventSource not initialized"));
        return;
      }

      // Use specific event listeners for each event type
      eventSource.addEventListener("connected", event => {
        try {
          const data = SSEEventDataSchema.parse(JSON.parse(event.data as string));
          if (data.message) {
            this.onConnected?.(data.message);
          }
        } catch (error) {
          console.error("Failed to parse connected event:", error);
          const errorMsg = "Failed to establish connection properly. Please try again.";
          this.onError?.(errorMsg);
          this.cleanup();
          reject(new Error(errorMsg));
        }
      });

      eventSource.addEventListener("progress", event => {
        try {
          const data = SSEEventDataSchema.parse(JSON.parse(event.data as string));
          if (data.message) {
            const progress = this.calculateProgress(data.message);
            this.onProgress?.(data.message, progress);
          }
        } catch (error) {
          console.error("Failed to parse progress event:", error);
          const errorMsg = "Connection issue during evaluation. Please try again.";
          this.onError?.(errorMsg);
          this.cleanup();
          reject(new Error(errorMsg));
        }
      });

      eventSource.addEventListener("complete", event => {
        try {
          const data = SSEEventDataSchema.parse(JSON.parse(event.data as string));
          if (data.result) {
            this.onComplete?.(data.result);
            this.cleanup();
            resolve(data.result);
          } else {
            this.cleanup();
            reject(new Error("Complete event missing result data"));
          }
        } catch (error) {
          console.error("Failed to parse complete event:", error);
          const errorMsg =
            "Evaluation completed but the response format was invalid. Please try again.";
          this.onError?.(errorMsg);
          this.cleanup();
          reject(new Error(errorMsg));
        }
      });

      eventSource.addEventListener("error", event => {
        try {
          const messageEvent = event as MessageEvent;
          const data = SSEEventDataSchema.parse(
            JSON.parse(messageEvent.data as string),
          );
          const errorMessage = data.message ?? "Unknown error occurred";
          this.onError?.(errorMessage);
          this.cleanup();
          reject(new Error(errorMessage));
        } catch (error) {
          console.error("Failed to parse error event:", error);
          const errorMsg = "Error occurred during evaluation. Please try again.";
          this.onError?.(errorMsg);
          this.cleanup();
          reject(new Error(errorMsg));
        }
      });

      eventSource.onerror = event => {
        // Try to parse error data if available
        try {
          const messageEvent = event as MessageEvent;
          if (messageEvent.data) {
            const data = SSEEventDataSchema.parse(
              JSON.parse(messageEvent.data as string),
            );
            const errorMessage = data.message ?? "Unknown error occurred";
            this.onError?.(errorMessage);
            this.cleanup();
            reject(new Error(errorMessage));
            return;
          }
        } catch (error) {
          console.error("Failed to parse error data:", error);
        }

        // Fallback to generic connection error
        this.onConnectionError?.(event);
        this.cleanup();
        reject(new Error("Connection lost"));
      };
    });
  }

  /**
   * Stop the current evaluation
   */
  stopEvaluation(): void {
    this.cleanupWithRejection("Evaluation cancelled by user");
  }

  /**
   * Check if evaluation is currently in progress
   */
  get isRunning(): boolean {
    return this.isEvaluating;
  }

  private cleanup(): void {
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
    }
    this.isEvaluating = false;
    this.currentReject = undefined;
  }

  private cleanupWithRejection(reason: string): void {
    if (this.currentReject) {
      this.currentReject(new Error(reason));
      this.currentReject = undefined;
    }
    this.cleanup();
  }

  private calculateProgress(message: string): number {
    const phases = [
      "Fetching arXiv data",
      "Fetching Semantic Scholar data and parsing arXiv paper",
      "Fetching Semantic Scholar references and recommendations",
      "Extracting annotations and classifying contexts",
      "Discovering related papers",
      "Generating related paper summaries",
      "Extracting graph representation",
      "Evaluating novelty",
    ];

    const currentPhase = phases.findIndex(phase =>
      message.toLowerCase().includes(phase.toLowerCase()),
    );

    if (currentPhase >= 0) {
      // Show progress for the current phase starting (not completing)
      // Don't immediately jump to 100% on the last phase
      if (currentPhase === phases.length - 1) {
        return 90; // Show 90% when final phase starts, reserve 100% for completion
      }
      return (currentPhase / phases.length) * 100 + (100 / phases.length) * 0.1;
    }

    return 0;
  }
}

/**
 * Input parameters for partial paper evaluation.
 */
export interface PartialEvaluationParams {
  title: string;
  abstract: string;
  recommendations?: number;
  llm_model?: string;
  related?: number;
}

/**
 * Real-time partial paper evaluator using Server-Sent Events.
 * Provides live progress updates during partial evaluation using only title and abstract.
 *
 * TODO: This duplicates significant logic from PaperEvaluator - consider refactoring
 * to share common SSE handling code in a future iteration.
 */
export class PartialPaperEvaluator {
  private eventSource: EventSource | null = null;
  private isEvaluating = false;
  private currentReject?: (reason?: unknown) => void;

  onConnected?: (message: string) => void;
  onProgress?: (message: string, progress: number) => void;
  onComplete?: (result: PartialEvaluationResponse) => void;
  onError?: (error: string) => void;
  onConnectionError?: (event: Event) => void;

  constructor(private baseUrl: string) {}

  /**
   * Start partial paper evaluation with real-time progress updates
   */
  async startEvaluation(
    paperData: PartialEvaluationParams,
  ): Promise<PartialEvaluationResponse> {
    if (this.isEvaluating) {
      throw new Error("Evaluation already in progress");
    }

    // Build URL parameters
    const params = new URLSearchParams({
      title: paperData.title,
      abstract: paperData.abstract,
    });

    if (paperData.recommendations) {
      params.set("recommendations", paperData.recommendations.toString());
    }
    if (paperData.llm_model) {
      params.set("llm_model", paperData.llm_model);
    }
    if (paperData.related) {
      params.set("related", paperData.related.toString());
    }

    const url = `${this.baseUrl}/mind/evaluate-partial?${params}`;
    this.eventSource = new EventSource(url);
    this.isEvaluating = true;

    return new Promise((resolve, reject) => {
      // Store reject function for cancellation
      this.currentReject = reject;

      const eventSource = this.eventSource;
      if (!eventSource) {
        reject(new Error("EventSource not initialized"));
        return;
      }

      // Use specific event listeners for each event type
      eventSource.addEventListener("connected", event => {
        try {
          const data = SSEEventDataSchema.parse(JSON.parse(event.data as string));
          if (data.message) {
            this.onConnected?.(data.message);
          }
        } catch (error) {
          console.error("Failed to parse connected event:", error);
          const errorMsg = "Failed to establish connection properly. Please try again.";
          this.onError?.(errorMsg);
          this.cleanup();
          reject(new Error(errorMsg));
        }
      });

      eventSource.addEventListener("progress", event => {
        try {
          const data = SSEEventDataSchema.parse(JSON.parse(event.data as string));
          if (data.message) {
            const progress = this.calculateProgress(data.message);
            this.onProgress?.(data.message, progress);
          }
        } catch (error) {
          console.error("Failed to parse progress event:", error);
          const errorMsg = "Connection issue during evaluation. Please try again.";
          this.onError?.(errorMsg);
          this.cleanup();
          reject(new Error(errorMsg));
        }
      });

      eventSource.addEventListener("complete", event => {
        try {
          const data = PartialSSEEventDataSchema.parse(
            JSON.parse(event.data as string),
          );
          if (data.result) {
            this.onComplete?.(data.result);
            this.cleanup();
            resolve(data.result);
          } else {
            this.cleanup();
            reject(new Error("Complete event missing result data"));
          }
        } catch (error) {
          console.error("Failed to parse complete event:", error);
          const errorMsg =
            "Evaluation completed but the response format was invalid. Please try again.";
          this.onError?.(errorMsg);
          this.cleanup();
          reject(new Error(errorMsg));
        }
      });

      eventSource.addEventListener("error", event => {
        try {
          const messageEvent = event as MessageEvent;
          const data = SSEEventDataSchema.parse(
            JSON.parse(messageEvent.data as string),
          );
          const errorMessage = data.message ?? "Unknown error occurred";
          this.onError?.(errorMessage);
          this.cleanup();
          reject(new Error(errorMessage));
        } catch (error) {
          console.error("Failed to parse error event:", error);
          const errorMsg = "Error occurred during evaluation. Please try again.";
          this.onError?.(errorMsg);
          this.cleanup();
          reject(new Error(errorMsg));
        }
      });

      eventSource.onerror = event => {
        // Try to parse error data if available
        try {
          const messageEvent = event as MessageEvent;
          if (messageEvent.data) {
            const data = SSEEventDataSchema.parse(
              JSON.parse(messageEvent.data as string),
            );
            const errorMessage = data.message ?? "Unknown error occurred";
            this.onError?.(errorMessage);
            this.cleanup();
            reject(new Error(errorMessage));
            return;
          }
        } catch (error) {
          console.error("Failed to parse error data:", error);
        }

        // Fallback to generic connection error
        this.onConnectionError?.(event);
        this.cleanup();
        reject(new Error("Connection lost"));
      };
    });
  }

  /**
   * Stop the current evaluation
   */
  stopEvaluation(): void {
    this.cleanupWithRejection("Evaluation cancelled by user");
  }

  /**
   * Check if evaluation is currently in progress
   */
  get isRunning(): boolean {
    return this.isEvaluating;
  }

  private cleanup(): void {
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
    }
    this.isEvaluating = false;
    this.currentReject = undefined;
  }

  private cleanupWithRejection(reason: string): void {
    if (this.currentReject) {
      this.currentReject(new Error(reason));
      this.currentReject = undefined;
    }
    this.cleanup();
  }

  private calculateProgress(message: string): number {
    const phases = [
      "Extracting keywords from abstract",
      "Searching related papers",
      "Extracting background and target from abstracts",
      "Retrieving semantic papers",
      "Summarising related papers",
      "Evaluating partial paper",
    ];

    const currentPhase = phases.findIndex(phase =>
      message.toLowerCase().includes(phase.toLowerCase()),
    );

    if (currentPhase >= 0) {
      // Show progress for the current phase starting (not completing)
      // Don't immediately jump to 100% on the last phase
      if (currentPhase === phases.length - 1) {
        return 90; // Show 90% when final phase starts, reserve 100% for completion
      }
      return (currentPhase / phases.length) * 100 + (100 / phases.length) * 0.1;
    }

    return 0;
  }
}
