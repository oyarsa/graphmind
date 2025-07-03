# Paper Explorer

A TypeScript web application for exploring research papers through two distinct
interfaces: an interactive network visualization and a searchable grid-based explorer
with detailed paper analysis.

## Features

### 🕸️ Paper Network

- **Interactive D3.js network visualization** for real-time paper exploration
- **Force-directed graph** with draggable nodes and dynamic expansion
- **Citation and semantic relationships** displayed as connections
- **Real-time API integration** for fetching paper data and relationships
- **Interactive info panel** for detailed paper information

### 📚 Paper Explorer

- **Searchable paper grid** by paper title, author and abstract
- **Dataset-driven browsing** of research papers
- **Detailed paper view** with comprehensive information display
- **Related papers analysis** showing background, context, supporting, and contrasting
  relationships
- **Keywords and metadata** display for each paper

## Technology Stack

- **TypeScript** with strict type checking and Zod validation
- **D3.js** for data visualization and SVG manipulation
- **Tailwind CSS** with minimal custom CSS for complex effects
- **Vite** for development server and bundling
- **Vitest** for unit testing

## Prerequisites

- Node.js (version 18 or higher recommended)
- [just](https://github.com/casey/just) command runner

## Installation

```bash
npm install
```

## Development

### Start Development Server

```bash
just dev
```

The application will be available at `http://localhost:5173`

### Available Commands

**Development:**

```bash
just dev        # Start development server
just build      # Build for production
just preview    # Preview production build
```

**Code Quality:**

```bash
just lint       # Run all checks: lint, format, type check, and tests
just check      # Check eslint only
just fix        # Fix eslint issues
just fmt        # Format with prettier
just type       # Run TypeScript type checker
just test       # Run unit tests
just check-all  # Check eslint and type checker
just watch      # Watch files and run checks on changes
```

## Project Structure

```
src/
├── paper-network/          # Interactive network visualization
│   ├── main.ts            # Application initialization
│   ├── network.ts         # D3.js PaperNetwork class
│   ├── apiService.ts      # API integration
│   └── model.ts           # Data models
├── paper-explorer/         # Grid-based paper browser
│   ├── main.ts            # Paper loading and search
│   ├── detail.ts          # Individual paper detail view
│   ├── model.ts           # Zod schemas
│   └── helpers.ts         # Utility functions
├── style.css              # Shared CSS styles
├── theme.ts               # Dark/light theme management
└── util.ts                # Shared utilities

pages/
├── paper-network.html     # Network visualization page
├── paper-explorer.html    # Grid browser page
└── paper-detail.html     # Individual paper detail page
```

## Usage

1. **Home Page** (`/`): Central landing page with navigation to both interfaces

2. **Paper Network** (`/pages/paper-network.html`):

   - Explore papers through an interactive force-directed graph
   - Click nodes to expand and see related papers
   - Drag nodes to rearrange the visualization
   - View paper details in the interactive panel

3. **Paper Explorer** (`/pages/paper-explorer.html`):
   - Browse papers in a searchable grid layout
   - Use fuzzy search to find specific papers
   - Click on papers to view detailed information
   - Explore related papers and their relationships

## Data Sources

- **Paper Network**: Real-time API queries for dynamic paper relationships
- **Paper Explorer**: Static JSON dataset loaded from environment-specified path

## Development Workflow

1. Make your changes
2. Run `just lint` to ensure code quality and tests pass
3. The lint command runs: ESLint fixes, Prettier formatting, TypeScript checking, and
   unit tests

## Testing

Unit tests are written using Vitest and can be run with:

```bash
just test
# or as part of the full lint process
just lint
```
