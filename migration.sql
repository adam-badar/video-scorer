-- Video Scorecards table for the video-scorer API.
-- Used by both the Azure Container Apps worker (writes) and
-- the Bavlio-Admin dashboard (reads via server-side API routes).
-- All access uses the Supabase service key — no anon key, no client-side reads.

CREATE TABLE IF NOT EXISTS video_scorecards (
    id               BIGSERIAL PRIMARY KEY,
    analysis_id      UUID UNIQUE NOT NULL,
    file_hash        TEXT,
    file_name        TEXT,
    video_url        TEXT,
    platform         TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'queued',
    error_message    TEXT,
    duration_seconds FLOAT8,
    resolution       TEXT,
    wpm              FLOAT8,
    filler_word_count INTEGER,
    cuts_per_minute  FLOAT8,
    loudness_lufs    FLOAT8,
    true_peak_dbtp   FLOAT8,
    silence_ratio    FLOAT8,
    loop_score       FLOAT8,
    hook_score       INTEGER,
    pacing_score     INTEGER,
    editing_score    INTEGER,
    audio_score      INTEGER,
    structure_score  INTEGER,
    total_score      INTEGER,
    max_possible     INTEGER,
    grade            TEXT,
    detected_language TEXT,
    language_warning TEXT,
    qualitative      JSONB,
    raw_scorecard    JSONB,
    scored_at        TIMESTAMPTZ,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

-- RLS enabled but no user-facing policies.
-- Service key bypasses RLS — only service-key access from
-- the Azure worker and Vercel API routes.
ALTER TABLE video_scorecards ENABLE ROW LEVEL SECURITY;

-- Index for polling by analysis_id (primary query pattern)
CREATE INDEX IF NOT EXISTS idx_video_scorecards_analysis_id ON video_scorecards (analysis_id);

-- Index for history queries (sorted by scored_at, filtered by status)
CREATE INDEX IF NOT EXISTS idx_video_scorecards_history ON video_scorecards (status, scored_at DESC);
