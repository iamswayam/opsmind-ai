-- Migration: add multimodal support to an EXISTING database.
-- db/init.sql only runs on first container boot, so this is needed separately
-- for any database that already has data in it.
--
-- Run with:
--   docker compose exec db psql -U postgres -d opsmind -f /migrations/001_multimodal.sql
-- (this file needs to be reachable inside the db container — see instructions
-- below if you'd rather just pipe it in directly instead of mounting it)

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS modality TEXT NOT NULL DEFAULT 'text';
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS image_data BYTEA;
