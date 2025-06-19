-- Create enum type for related link types
DO $$ BEGIN
    CREATE TYPE related_type AS ENUM ('semantic', 'citation');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Create the Paper table
CREATE TABLE IF NOT EXISTS paper (
    id VARCHAR PRIMARY KEY,
    title TEXT NOT NULL,
    year INTEGER NOT NULL,
    authors TEXT[] NOT NULL,
    abstract TEXT NOT NULL,
    venue TEXT NOT NULL,
    citation_count INTEGER NOT NULL,
    doi VARCHAR,
    pdf_url VARCHAR
);

-- Create the Related table
CREATE TABLE IF NOT EXISTS related (
    source VARCHAR NOT NULL,
    target VARCHAR NOT NULL,
    type related_type NOT NULL,
    similarity FLOAT NOT NULL,
    PRIMARY KEY (source, target, type),
    FOREIGN KEY (source) REFERENCES paper(id) ON DELETE CASCADE,
    FOREIGN KEY (target) REFERENCES paper(id) ON DELETE CASCADE
);

-- Add indexes for foreign keys
CREATE INDEX IF NOT EXISTS idx_related_source ON related(source);
CREATE INDEX IF NOT EXISTS idx_related_target ON related(target);

-- Add individual full-text search indexes
CREATE INDEX IF NOT EXISTS idx_paper_title_gin ON paper USING GIN (to_tsvector('english', title));
CREATE INDEX IF NOT EXISTS idx_paper_abstract_gin ON paper USING GIN (to_tsvector('english', abstract));

-- Add the weighted tsvector column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='paper' AND column_name='search_vector') THEN
        ALTER TABLE paper ADD COLUMN search_vector tsvector
            GENERATED ALWAYS AS (
                setweight(to_tsvector('english', title), 'A') ||
                setweight(to_tsvector('english', abstract), 'B')
            ) STORED;
    END IF;
END $$;

-- Create index on search_vector
CREATE INDEX IF NOT EXISTS idx_paper_search_vector ON paper USING GIN (search_vector);
