-- Create application roles
CREATE APPLICATION ROLE project_assistant_admin;
CREATE APPLICATION ROLE project_assistant_service;
GRANT APPLICATION ROLE project_assistant_service TO APPLICATION ROLE admin;

-- Create versioned schemas for app code and state
CREATE OR REPLACE VERSIONED SCHEMA pa_code;
CREATE OR REPLACE SCHEMA pa_core;

-- Grant usage on schemas
GRANT USAGE ON SCHEMA pa_code TO APPLICATION ROLE project_assistant_admin;
GRANT USAGE ON SCHEMA pa_code TO APPLICATION ROLE project_assistant_service;
GRANT USAGE ON SCHEMA pa_core TO APPLICATION ROLE project_assistant_admin;
GRANT USAGE ON SCHEMA pa_core TO APPLICATION ROLE project_assistant_service;

-- Create tables for metadata storage
CREATE OR REPLACE HYBRID TABLE pa_core.projects (
    project_id VARCHAR NOT NULL,
    name VARCHAR,
    description VARCHAR,
    pinned_documents ARRAY,
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    updated_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    config VARIANT,
    PRIMARY KEY (project_id)
);

CREATE OR REPLACE HYBRID TABLE pa_core.meetings (
    meeting_id VARCHAR NOT NULL,
    project_id VARCHAR NOT NULL,
    title VARCHAR,
    status VARCHAR,
    participants ARRAY,
    related_documents ARRAY,
    metadata VARIANT,
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    updated_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (meeting_id),
    FOREIGN KEY (project_id) REFERENCES pa_core.projects(project_id)
);

CREATE OR REPLACE HYBRID TABLE pa_core.documents (
    content_id VARCHAR NOT NULL,
    project_id VARCHAR NOT NULL,
    source_type VARCHAR NOT NULL,
    title VARCHAR,
    content_hash VARCHAR,
    version INTEGER,
    metadata VARIANT,
    token_count INTEGER,
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    updated_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (content_id, project_id, source_type),
    FOREIGN KEY (project_id) REFERENCES pa_core.projects(project_id)
);

-- Grant privileges on tables
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE pa_core.projects TO APPLICATION ROLE project_assistant_admin;
GRANT SELECT ON TABLE pa_core.projects TO APPLICATION ROLE project_assistant_service;

GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE pa_core.meetings TO APPLICATION ROLE project_assistant_admin;
GRANT SELECT ON TABLE pa_core.meetings TO APPLICATION ROLE project_assistant_service;

GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE pa_core.documents TO APPLICATION ROLE project_assistant_admin;
GRANT SELECT ON TABLE pa_core.documents TO APPLICATION ROLE project_assistant_service;
