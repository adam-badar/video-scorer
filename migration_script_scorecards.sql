-- Script scorecards table for pre-filming script analysis
-- Simplified schema: synchronous scoring, no polling state machine

CREATE TABLE IF NOT EXISTS script_scorecards (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    platform TEXT NOT NULL,
    script_text TEXT NOT NULL,
    grade TEXT,
    total_score INTEGER,
    max_possible INTEGER,
    qualitative JSONB,
    raw_scorecard JSONB,
    scored_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- RLS: service-key-only access (no user-facing policies)
ALTER TABLE script_scorecards ENABLE ROW LEVEL SECURITY;

-- Index for history queries
CREATE INDEX IF NOT EXISTS idx_script_scorecards_history
    ON script_scorecards (scored_at DESC);
