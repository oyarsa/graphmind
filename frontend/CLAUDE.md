# Codebase Guidelines

This file provides guidance to AI agents when working with code in this repository.

## Important Workflow Rules

**ALWAYS run `just lint` after making any code changes before considering the task
complete.** This ensures code formatting, type checking, and all tests pass.

**Run `just e2e` after major frontend/API changes** to run the end-to-end Playwright tests.
The servers are started automatically by Playwright.

**ALWAYS notify the user when you are done, even if you don't need input from them.**

## Development Commands

**Start development server:**

```bash
just dev
```

**Build project:**

```bash
just build
```

**Type checking:**

```bash
just type
```

**Linting and Formatting:**

```bash
just check      # Check eslint only
just fix        # Fix eslint issues only
just fmt        # Format with prettier
just lint       # Fix eslint, format, type check and tests
just check-all  # Check eslint, type check and tests
```

**Watch for changes:**

```bash
just watch      # Watch files and run check-all on changes
```

**Preview built project:**

```bash
just preview
```

## Architecture Overview

This is a TypeScript web application for exploring research papers through two main interfaces.
The application uses Vite as the build tool and follows a clean, modular architecture pattern.

**Key Technologies:**

- TypeScript with strict type checking and Zod validation
- D3.js for data visualization and SVG manipulation
- Tailwind CSS with minimal custom CSS for complex effects
- Vite for development and bundling

**Main Components:**

The application consists of two primary interfaces accessible from a central home page:

1. **Paper Network** (`/pages/paper-network.html`): Interactive D3.js network visualization

   - **Purpose**: Real-time paper exploration through network graphs
   - **Architecture**: `src/paper-network/`
     - `main.ts`: Application initialization with API service
     - `network.ts`: Core D3.js-based `PaperNetwork` class for graph visualization
     - `apiService.ts`: API integration for fetching paper data and relationships
     - `model.ts`: Data models and type definitions
   - **Features**: Force-directed graph, draggable nodes, citation/semantic relationships,
     interactive info panel, dynamic node expansion

2. **Paper Explorer** (`/pages/paper-explorer.html`): Grid-based paper browsing with detail views

   - **Purpose**: Dataset-driven paper discovery and detailed analysis
   - **Architecture**: `src/paper-explorer/`
     - `main.ts`: Paper loading, search, grid display
     - `detail.ts`: Individual paper detail sub-page with related papers analysis
     - `model.ts`: Zod schemas for data validation
   - **Data Source**: Static JSON dataset loaded from environment-specified path
   - **Features**:
     - **Main View**: Searchable paper grid with fuzzy search
     - **Detail Sub-page** (`/pages/paper-detail.html`): Individual paper view with full information,
       keywords, and related papers breakdown (background, context, supporting, contrasting relationships)

**Data Flow:**

- **Network**: API queries → Real-time paper relationships → Dynamic graph updates
- **Explorer**: JSON dataset → Fuzzy search filtering → Grid display → Detail navigation
- **Explorer Detail**: URL paper ID → localStorage dataset lookup → Related papers analysis

**File Structure:**

- `index.html`: Central landing page with navigation to both main interfaces
- `pages/`: HTML pages for each interface (network, explorer, detail sub-page)
- `src/paper-network/`: Network visualization components
- `src/paper-explorer/`: Grid-based explorer and detail sub-page components
- `src/style.css`: Shared CSS for complex effects (backdrop filters, gradients, scrollbars)
- `src/theme.ts`: Dark/light theme management
- `src/util.ts`: Shared utilities (DOM helpers, error handling, mobile detection)

**Key Differences:**

- **Paper Network**: Real-time API-driven network visualization for live exploration
- **Paper Explorer**: Dataset-driven grid interface with detailed paper analysis sub-pages

## Version control

- Use jujutsu (jj) instead of git to manage version control in this repository.
- After each major operation, create a new revision.
- In general, follow all the common git workflows, but using jujutsu instead. If you
  don't know how to do something, ask and I'll help you.
- When creating commits for the frontend, prefix all messages with `frontend: {msg}`

## End-to-End Testing with Playwright

### Automated E2E Tests

The `tests/e2e/` directory contains automated Playwright tests that verify the full
evaluation flow. Tests run in parallel (up to 5 papers concurrently) for speed.

**Setup (first time only):**

```bash
npm install                         # Install dependencies including Playwright
just e2e-install                    # Install Playwright browsers
```

**Running the tests:**

```bash
just e2e                          # Run all e2e tests (headless)
just e2e-headed                   # Run with browser visible (for debugging)
```

Servers are started automatically by Playwright's `webServer` config. If you already have
the servers running, they will be reused (set `reuseExistingServer: false` in
`playwright.config.ts` to force fresh servers).

**What the tests check:**

- Search page loads correctly
- arXiv search returns results
- Paper evaluation completes successfully
- Detail page shows correct data (year, novelty score, evidence, graph)
- Evidence has no raw LaTeX commands or bad summary patterns
- Evidence distribution (max 5 per type, semantic capped at 3)
- Date filtering (related papers not from the future)
- Sections are collapsible
- Related papers filtering works
- Cached papers navigate directly to detail

### Manual Testing with Playwright MCP

Use the Playwright MCP tools for interactive/exploratory testing.

**Setup:**

1. Create `.env` with required variables:

   ```bash
   VITE_API_URL=http://localhost:8000
   VITE_XP_DATA_PATH=data/test-data.json
   ```

2. Start both servers (in background):
   ```bash
   just api-dev &                    # Backend on :8000 (from repo root)
   just dev &                        # Frontend on :5173
   ```

**Test flow using Playwright MCP:**

1. Navigate to `http://localhost:5173/graphmind/`
2. Click "arXiv" tab
3. Search for a paper (e.g., "Weak Reward Model")
4. Click the paper card to open evaluation settings
5. Select LLM model (e.g., "gpt-4o-mini")
6. Click "Start Evaluation"
7. Wait for progress modal to complete (~2-3 minutes)
8. Verify the detail page shows:
   - Novelty score (1-5)
   - Paper Structure Graph
   - Supporting and contradictory evidence
   - Related papers

**Key Playwright MCP tools:**

- `browser_navigate` - Go to URL
- `browser_click` - Click elements by ref
- `browser_type` - Type in inputs (use `submit: true` to press Enter)
- `browser_select_option` - Select dropdown options
- `browser_snapshot` - Get page state (accessibility tree)
- `browser_take_screenshot` - Capture visual state
- `browser_console_messages` - Check for errors
- `browser_wait_for` - Wait for time or text

**Cleanup after testing:**

```bash
lsof -ti:8000 | xargs -r kill -9    # Kill backend
lsof -ti:5173 | xargs -r kill -9    # Kill frontend
rm .env                              # Remove test config
```

## Memories

- Don't ask to run a server since it's an interactive thing that you can't really use. Just ask the user to run it.
- **IMPORTANT**: Minimise the use of custom CSS. Use Tailwind for everything as much as possible.
- Use `just test` to run unit tests, or run them through `just lint` with everything else.
- All features should be supported cross-browser, not only Chrome and WebKit. Firefox is an important browser for us.
