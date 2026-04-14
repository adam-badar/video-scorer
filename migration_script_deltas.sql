-- Script deltas table for three-way comparison (AI draft vs final script vs spoken transcript)

CREATE TABLE IF NOT EXISTS script_deltas (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    video_scorecard_analysis_id UUID NOT NULL,
    platform TEXT NOT NULL,
    ai_draft_text TEXT,
    final_script_text TEXT NOT NULL,
    delta_analysis JSONB NOT NULL,
    voice_example_content TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT fk_video_scorecard
        FOREIGN KEY (video_scorecard_analysis_id)
        REFERENCES video_scorecards(analysis_id)
);

-- RLS: service-key-only access (no user-facing policies)
ALTER TABLE script_deltas ENABLE ROW LEVEL SECURITY;

-- Index for listing deltas by scorecard
CREATE INDEX IF NOT EXISTS idx_script_deltas_scorecard
    ON script_deltas (video_scorecard_analysis_id);

-- Index for history queries
CREATE INDEX IF NOT EXISTS idx_script_deltas_created
    ON script_deltas (created_at DESC);
