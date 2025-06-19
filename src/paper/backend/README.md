# Paper Explorer Backend

A FastAPI-based backend service for exploring academic papers and their relationships
through citations and semantic similarity.

## Features

- **Paper Search**: Full-text search across paper titles and abstracts using
  PostgreSQL's text search capabilities.
- **Citation Network**: Explore citation relationships between papers.
- **Semantic Similarity**: Discover papers with similar content through semantic
  relationships with pgvector.
- **RESTful API**: REST endpoints for all operations.

## Architecture

- **FastAPI**: Modern Python web framework with automatic OpenAPI documentation.
- **PostgreSQL**: Primary database with full-text search and pgvector extension.
- **Docker Compose**: Containerised PostgreSQL setup for easy development.
- **Pydantic**: Data validation and serialisation.
- **Static Typing**: Full type coverage with pyright.

## Database Schema

The system uses two main tables:

- **paper**: Stores paper metadata (title, authors, abstract, venue, citation count,
  etc.)
- **related**: Stores relationships between papers (citations and semantic similarity)

## Getting Started

1. Clone the repository and navigate to the backend directory:
```bash
cd paper-explorer/backend
```

2. Install dependencies using uv:
```bash
uv sync
```
If you don't have uv installed, install it first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Start the PostgreSQL database:
```bash
docker compose up -d
```
**Note**: Ensure Docker is running before executing this command.

4. Set up environment variables by copying the example file:
```bash
cp .env.example .env
```

The default values should work for local development:
```
XP_DB_NAME=explorer
XP_DB_USER=postgres
XP_DB_PASSWORD=dev
XP_DB_HOST=localhost
XP_DB_PORT=5432
```

### Database Setup and Data Generation

1. Start the PostgreSQL database:
```bash
docker compose up -d
```
**Note**: Ensure Docker is running before executing this command.

2. Set up environment variables by copying the example file:
```bash
# In the repository root.
cp .env.example .env
```

The default values should work for local development:
```
XP_DB_NAME=explorer
XP_DB_USER=postgres
XP_DB_PASSWORD=dev
XP_DB_HOST=localhost
XP_DB_PORT=5432
```

3. Initialise the database extensions:
```bash
uv run -m paper.backend.db.execute sql/init.sql
```

4. Create database schema:
```bash
uv run -m paper.backend.db.execute sql/basic_schema.sql
```

5. Generate synthetic test data (this may take a few minutes):
```bash
uv run -m paper.backend.generate_data data/papers.json --num-papers 5000
```
**Note**: You can adjust `--num-papers` for different dataset sizes. Larger datasets
will take longer to generate.

6. Load the generated data into the database:
```bash
uv run -m paper.backend.db.seed data/papers.json
```

### Running the Application

#### Development Server
```bash
just dev
```
The API will be available at `http://localhost:8000` with automatic reload on code
changes.

#### Production Server
```bash
just serve
```
The API will be available at `http://localhost:8001`.

### API Documentation

Once the server is running, visit:
- Interactive API docs: `http://localhost:8000/docs`
- ReDoc documentation: `http://localhost:8000/redoc`

## API Endpoints

### Health Check
- `GET /health` - Health check endpoint

### Paper Search and Retrieval (Network)
- `GET /network/search?q=query&limit=5` - Search papers by title/abstract
- `GET /network/papers/{id}` - Get paper details by ID
- `GET /network/related/{paper_id}?type=citation&limit=5` - Get related papers

### Paper Analysis and Evaluation (Mind)
- `GET /mind/search?q=query&limit=5` - Search arXiv papers by title
- `GET /mind/evaluate?id=arxiv_id&title=paper_title` - Comprehensive paper analysis

#### Mind Evaluation Parameters
- `id` - arXiv ID of the paper to analyse
- `title` - Title of the paper on arXiv
- `k_refs` - Number of references to analyse (1-10, default: 2)
- `recommendations` - Number of recommended papers (5-50, default: 30)
- `related` - Number of related papers per type (1-10, default: 2)
- `llm_model` - LLM model to use (`gpt-4o`, `gpt-4o-mini`, `gemini-2.0-flash`)
- `seed` - Random seed for reproducible results (default: 0)

### Relationship Types
- `citation` - Papers connected through citation relationships
- `semantic` - Papers with similar content

## Development

### Database Management

#### Reset Test Database

Drop the existing tables, re-generate and insert test data.

```bash
./dev/reset_test_db.sh
```

#### Generate New Test Data
```bash
uv run python -m paper.backend.generate_data data/papers.json --num-papers 1000 --seed 42
```

## Environment Variables

| Variable                  | Description                             | Default     | Required |
|---------------------------|-----------------------------------------|-------------|----------|
| `XP_DB_NAME`              | PostgreSQL database name                | `explorer`  | Yes      |
| `XP_DB_USER`              | PostgreSQL username                     | `postgres`  | Yes      |
| `XP_DB_PASSWORD`          | PostgreSQL password                     | `dev`       | Yes      |
| `XP_DB_HOST`              | PostgreSQL host                         | `localhost` | Yes      |
| `XP_DB_PORT`              | PostgreSQL port                         | `5432`      | Yes      |
| `API_ENV`                 | API environment (`production` or other) |             | No       |
| `ALLOWED_ORIGINS`         | CORS allowed origins (production only)  |             | Prod only|
| `SEMANTIC_SCHOLAR_API_KEY`| Semantic Scholar API key for paper data |             | No       |
| `OPENAI_API_KEY`          | OpenAI API key for LLM evaluation       |             | No       |
| `OPENAI_API_TIER`         | OpenAI API tier level                   | `1`         | No       |
