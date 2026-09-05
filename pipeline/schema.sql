-- ==============================================================================
-- Supabase pgvector Schema for Document Embeddings
-- Run this script in the Supabase SQL Editor (Dashboard -> SQL Editor -> New Query)
-- ==============================================================================

-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create the document_embeddings table
-- Note: Set vector dimension to 768 (standard for Gemini output_dimensionality=768)
-- If using full 3072 dimensions, update vector(768) to vector(3072).
CREATE TABLE IF NOT EXISTS "NILA_embeddings" (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding VECTOR(768),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 3. Create HNSW index for fast approximate nearest neighbor search
CREATE INDEX IF NOT EXISTS nila_embeddings_hnsw_idx
ON "NILA_embeddings"
USING hnsw (embedding vector_cosine_ops);

-- 4. Create RPC similarity search function
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding VECTOR(768),
    match_count INT DEFAULT 5,
    filter JSONB DEFAULT '{}'::jsonb
)
RETURNS TABLE (
    id BIGSERIAL,
    content TEXT,
    metadata JSONB,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        "NILA_embeddings".id,
        "NILA_embeddings".content,
        "NILA_embeddings".metadata,
        1 - ("NILA_embeddings".embedding <=> query_embedding) AS similarity
    FROM "NILA_embeddings"
    WHERE (filter = '{}'::jsonb OR "NILA_embeddings".metadata @> filter)
    ORDER BY "NILA_embeddings".embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
