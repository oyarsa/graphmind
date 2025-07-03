# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this
repository.

## Important Workflow Rules

**ALWAYS run `just lint` after making any code changes before considering the task
complete.** This ensures code formatting, type checking, and all tests pass.

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

## Memories

- Don't ask to run a server since it's an interactive thing that you can't really use. Just ask the user to run it.
- **IMPORTANT**: Minimise the use of custom CSS. Use Tailwind for everything as much as possible.
- Use `just test` to run unit tests, or run them through `just lint` with everything else.
- All features should be supported cross-browser, not only Chrome and WebKit. Firefox is an important browser for us.
