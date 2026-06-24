"""Application configuration.

Keep config small and explicit. Do not read environment variables directly in agents.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = Field(default="local", validation_alias="APP_ENV")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    fireworks_api_key: str | None = Field(default=None, validation_alias="FIREWORKS_API_KEY")
    fireworks_model: str = Field(
        default="accounts/fireworks/models/deepseek-v4-pro",
        validation_alias="FIREWORKS_MODEL",
    )
    fireworks_base_url: str = Field(
        default="https://api.fireworks.ai/inference/v1/chat/completions",
        validation_alias="FIREWORKS_BASE_URL",
    )
    fireworks_max_tokens: int = Field(default=131072, ge=1, validation_alias="FIREWORKS_MAX_TOKENS")
    fireworks_top_k: int = Field(default=40, ge=0, validation_alias="FIREWORKS_TOP_K")
    fireworks_presence_penalty: float = Field(
        default=0.0,
        validation_alias="FIREWORKS_PRESENCE_PENALTY",
    )
    fireworks_frequency_penalty: float = Field(
        default=0.0,
        validation_alias="FIREWORKS_FREQUENCY_PENALTY",
    )

    langsmith_api_key: str | None = Field(default=None, validation_alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(
        default="multi-agent-research-lab",
        validation_alias="LANGSMITH_PROJECT",
    )

    tavily_api_key: str | None = Field(default=None, validation_alias="TAVILY_API_KEY")

    max_iterations: int = Field(default=6, ge=1, le=20, validation_alias="MAX_ITERATIONS")
    timeout_seconds: int = Field(default=60, ge=5, le=600, validation_alias="TIMEOUT_SECONDS")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""

    return Settings()
