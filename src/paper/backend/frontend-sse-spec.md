# Paper Evaluation API - Real-Time Progress with Server-Sent Events

## Overview

The `/mind/evaluate` endpoint provides real-time progress updates using Server-Sent
Events (SSE). This allows the frontend to display live progress updates and prevent the
UI from freezing during long-running paper evaluations.

## Endpoint Details

**URL:** `GET /mind/evaluate`
**Response Type:** `text/event-stream` (Server-Sent Events)
**Authentication:** None required yet

### Query Parameters

| Parameter         | Type    | Required | Default      | Range     | Description                               |
|-------------------|---------|----------|--------------|-----------|-------------------------------------------|
| `id`              | string  | ✅       | -            | -         | arXiv ID of the paper to analyse          |
| `title`           | string  | ✅       | -            | -         | Title of the paper on arXiv               |
| `k_refs`          | integer | ❌       | 2            | 1-10      | Number of references to analyse           |
| `recommendations` | integer | ❌       | 30           | 5-50      | Number of recommended papers to retrieve  |
| `related`         | integer | ❌       | 2            | 1-10      | Number of related papers per type         |
| `llm_model`       | string  | ❌       | "gpt-4o-mini"| See below | LLM model to use                          |
| `seed`            | integer | ❌       | 0            | -         | Random seed for reproducible results      |

### LLM Model Options
- `"gpt-4o"`
- `"gpt-4o-mini"`
- `"gemini-2.0-flash"`

## Event Types

The SSE stream now uses proper event types that can be handled with
`addEventListener()`. Each event has an `event:` field followed by JSON data:

```
event: connected
data: {"message": "Starting evaluation..."}

event: progress
data: {"message": "Fetching arXiv data"}

event: complete
data: {"result": {...}}

event: error
data: {"message": "Paper not found on arXiv: invalid_id"}
```

### TypeScript Interface
```typescript
interface SSEEventData {
  message?: string;
  result?: EvalResult;
}
```

### 1. Connected Event
Sent immediately when connection is established.
```
event: connected
data: {"message": "Starting evaluation..."}
```

### 2. Progress Events
Sent during each phase of processing (8 total phases).
```
event: progress
data: {"message": "Fetching arXiv data"}
```

### 3. Complete Event
Sent when processing finishes successfully.
```
event: complete
data: {
  "result": {
    "result": {
      "paper": { /* PaperResult object */ },
      "graph": { /* GraphResult object */ },
      "related": [ /* RelatedPaper array */ ]
      /* ... other fields */
    },
    "cost": 0.0123456789
  }
}
```

### 4. Error Event
Sent if an error occurs during processing. The connection is gracefully closed after this event.
```
event: error
data: {"message": "Paper not found on arXiv: invalid_id"}
```

**Important**: Errors are sent as proper SSE events, not as connection errors. The
EventSource `onerror` handler should only be used for network/connection issues, not for
application errors.

## Progress Phases

The following 8 progress messages are sent in order:

1. **"Fetching arXiv data"** - Retrieving paper from arXiv
2. **"Fetching Semantic Scholar data and parsing arXiv paper"** - Getting S2 metadata
   and parsing LaTeX
3. **"Fetching Semantic Scholar references and recommendations"** - Getting reference
   data and recommended papers
4. **"Extracting annotations and classifying contexts"** - Running GPT analysis on paper
   content
5. **"Discovering related papers"** - Finding related papers via citations and semantics
6. **"Generating related paper summaries"** - Creating summaries for related papers
7. **"Extracting graph representation"** - Building knowledge graph from paper
8. **"Evaluating novelty"** - Final novelty assessment

## UI Recommendations

### Progress Visualization
- **Progress Bar**: 0-100% based on phase completion (8 phases total)
- **Status Text**: Display current phase message
- **Phase Icons**: Different icons for each phase type
- **Estimated Time**: Optional time estimates based on phase

### User Experience
- **Cancel Button**: Allow users to stop evaluation (`evaluator.stopEvaluation()`)
- **Error Handling**: Graceful error messages with retry options
- **Connection Recovery**: Handle network interruptions
- **Loading States**: Show appropriate loading indicators
- **Automatic Cancellation**: Server automatically cancels processing when client
  disconnects

### Example UI Flow
```
┌─────────────────────────────────────────┐
│ Paper Evaluation in Progress...         │
│                                         │
│ ████████████░░░░░░░░░░░░░░░░░░░░ 62%    │
│                                         │
│ 🔗 Discovering related papers          │
│                                         │
│ [Cancel Evaluation]                     │
└─────────────────────────────────────────┘
```

## Error Handling

### Error Types

**Application Errors** (sent as SSE `error` events):
- `"Paper not found on arXiv: {id}"` - Invalid arXiv ID
- `"Paper not found on Semantic Scholar: {title}"` - S2 lookup failed
- `"Failed to download LaTeX for arXiv ID: {id}"` - LaTeX parsing failure
- `"No recommended papers found"` - No recommendations available

**Connection Errors** (trigger EventSource `onerror`):
- Network connectivity issues
- Server unavailable
- Timeout on connection establishment

### Recovery Strategies
- **Retry Button**: Allow users to retry failed evaluations
- **Parameter Adjustment**: Suggest reducing `k_refs`, `recommendations`, or `related`
- **Alternative Models**: Try different LLM models if one fails
- **Graceful Degradation**: Show partial results if available
