from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "CampusMind AI"
    database_path: Path = Path("storage/campusmind.sqlite3")
    index_dir: Path = Path("storage/index")
    dataset_path: Path = Path("data/campusmind_dataset.xlsx")
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2:3b"
    embedding_model: str = "BAAI/bge-m3"
    jwt_secret: str = Field(default="change-this-local-secret")
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440
    top_k_semantic: int = 10
    top_k_keyword: int = 10
    hybrid_top_k: int = 10
    final_context_k: int = 3

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    def ensure_storage(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_storage()
    return settings
