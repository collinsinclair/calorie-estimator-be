from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Calorie Estimation Service"

    # Add your API keys and other settings here
    OPENAI_API_KEY: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
