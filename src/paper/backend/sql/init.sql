-- Enable pgvector for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

CREATE EXTENSION IF NOT EXISTS pg_trgm; -- For fuzzy text matching
CREATE EXTENSION IF NOT EXISTS unaccent; -- For accent-insensitive search
