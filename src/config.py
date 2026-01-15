"""
Application configuration using pydantic-settings.

Loads configuration from environment variables with .env file support.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # SEC EDGAR Download
    sec_company_name: str = Field(
        default="AgenticRAG-10K-Project",
        description="Company name for SEC user-agent",
    )
    sec_email: str = Field(
        default="your.email@example.com",
        description="Email for SEC user-agent",
    )
    sec_output_dir: str = Field(
        default="./data/raw_10k",
        description="Output directory for downloaded 10-K filings",
    )

    # Ollama Configuration
    ollama_base_url: str = Field(
        default="http://127.0.0.1:11434",
        description="Ollama API base URL",
    )
    ollama_llm_model: str = Field(
        default="llama3.1:8b",
        description="LLM model for chat/agent",
    )
    ollama_embedding_model: str = Field(
        default="nomic-embed-text",
        description="Embedding model for vector search",
    )
    embedding_dimension: int = Field(
        default=768,
        description="Embedding dimension (must match model)",
    )
    llm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="LLM temperature",
    )
    llm_max_tokens: int = Field(
        default=2048,
        gt=0,
        description="Max tokens for LLM responses",
    )

    # Qdrant Configuration
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant server URL",
    )
    qdrant_collection_name: str = Field(
        default="sp500_10k_chunks",
        description="Qdrant collection name",
    )
    qdrant_distance_metric: Literal["Cosine", "Euclid", "Dot"] = Field(
        default="Cosine",
        description="Vector distance metric",
    )

    # Ingestion Pipeline
    parallel_workers: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Default number of parallel workers for indexing",
    )
    chunk_size: int = Field(
        default=600,
        gt=0,
        description="Chunk size in tokens",
    )
    chunk_overlap: int = Field(
        default=100,
        ge=0,
        description="Chunk overlap in tokens",
    )
    min_chunk_length: int = Field(
        default=50,
        ge=0,
        description="Minimum chunk length to keep",
    )
    embedding_batch_size: int = Field(
        default=32,
        gt=0,
        description="Batch size for embedding generation",
    )
    qdrant_batch_size: int = Field(
        default=100,
        gt=0,
        description="Batch size for Qdrant upserts",
    )

    # Redis Configuration
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL",
    )
    redis_queue_name: str = Field(
        default="indexer",
        description="Redis queue name for indexing tasks",
    )

    # API Server
    api_host: str = Field(
        default="0.0.0.0",
        description="FastAPI host",
    )
    api_port: int = Field(
        default=8000,
        gt=0,
        le=65535,
        description="FastAPI port",
    )
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Environment name",
    )
    debug: bool = Field(
        default=True,
        description="Enable debug mode",
    )
    api_key: str = Field(
        default="",
        description="API key for authentication (empty to disable)",
    )

    # Concurrency & Rate Limiting
    max_concurrent_agents: int = Field(
        default=8,
        gt=0,
        description="Maximum concurrent agent runs",
    )
    rate_limit_per_minute: int = Field(
        default=10,
        gt=0,
        description="Rate limit requests per minute per IP",
    )
    request_timeout: int = Field(
        default=120,
        gt=0,
        description="Request timeout in seconds",
    )

    # Phoenix Observability
    phoenix_enabled: bool = Field(
        default=True,
        description="Enable Phoenix tracing",
    )
    phoenix_collector_endpoint: str = Field(
        default="http://localhost:6006",
        description="Phoenix collector endpoint",
    )
    phoenix_project_name: str = Field(
        default="financial-10k-agentic-rag",
        description="Phoenix project name",
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Log level",
    )
    log_format: Literal["json", "text"] = Field(
        default="text",
        description="Log format",
    )
    log_file: str = Field(
        default="",
        description="Log file path (empty for stdout only)",
    )

    # Retrieval
    top_k: int = Field(
        default=5,
        gt=0,
        description="Number of chunks to retrieve per query",
    )
    min_similarity_score: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold",
    )
    hybrid_search_enabled: bool = Field(
        default=True,
        description="Enable hybrid search (vector + keyword)",
    )
    keyword_search_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Keyword search weight for hybrid search",
    )

    # Agent
    max_agent_iterations: int = Field(
        default=10,
        gt=0,
        description="Maximum ReAct loop iterations",
    )
    agent_verbose: bool = Field(
        default=True,
        description="Enable verbose agent logging",
    )

    @field_validator("sec_email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Basic email format validation."""
        if v and "@" not in v:
            raise ValueError("Invalid email format")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        """Ensure overlap is less than chunk size."""
        chunk_size = info.data.get("chunk_size", 600)
        if v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings: Application settings loaded from environment.
    """
    return Settings()
