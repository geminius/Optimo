-- Initialize database for robotics optimization platform

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create tables for model metadata
CREATE TABLE IF NOT EXISTS models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    framework VARCHAR(50) NOT NULL,
    size_mb FLOAT NOT NULL,
    parameters BIGINT NOT NULL,
    file_path TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    tags TEXT[],
    metadata JSONB
);

-- Create tables for optimization sessions
CREATE TABLE IF NOT EXISTS optimization_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID REFERENCES models(id) ON DELETE CASCADE,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    criteria JSONB NOT NULL,
    plan JSONB,
    results JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Create tables for optimization steps
CREATE TABLE IF NOT EXISTS optimization_steps (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES optimization_sessions(id) ON DELETE CASCADE,
    step_name VARCHAR(255) NOT NULL,
    agent_type VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    config JSONB,
    results JSONB,
    error_message TEXT,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create tables for evaluation reports
CREATE TABLE IF NOT EXISTS evaluation_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID REFERENCES models(id) ON DELETE CASCADE,
    session_id UUID REFERENCES optimization_sessions(id) ON DELETE CASCADE,
    report_type VARCHAR(50) NOT NULL,
    benchmarks JSONB NOT NULL,
    metrics JSONB NOT NULL,
    comparison_baseline JSONB,
    validation_status VARCHAR(50) NOT NULL,
    recommendations TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_models_created_at ON models(created_at);
CREATE INDEX IF NOT EXISTS idx_models_model_type ON models(model_type);
CREATE INDEX IF NOT EXISTS idx_optimization_sessions_status ON optimization_sessions(status);
CREATE INDEX IF NOT EXISTS idx_optimization_sessions_created_at ON optimization_sessions(created_at);
CREATE INDEX IF NOT EXISTS idx_optimization_steps_session_id ON optimization_steps(session_id);
CREATE INDEX IF NOT EXISTS idx_optimization_steps_status ON optimization_steps(status);
CREATE INDEX IF NOT EXISTS idx_evaluation_reports_model_id ON evaluation_reports(model_id);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_models_updated_at BEFORE UPDATE ON models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_optimization_sessions_updated_at BEFORE UPDATE ON optimization_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert default configuration
INSERT INTO models (name, version, model_type, framework, size_mb, parameters, file_path, tags)
VALUES ('system_config', '1.0.0', 'config', 'system', 0, 0, '/config/default.json', ARRAY['system', 'config'])
ON CONFLICT DO NOTHING;