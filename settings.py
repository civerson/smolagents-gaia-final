import base64
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import HttpUrl, SecretStr


class Settings(BaseSettings):
    def __init__(self):
        super().__init__()
        self.set_langfuse_auth()
    model_config = SettingsConfigDict(env_file='.env')
    scoring_api_base_url: HttpUrl = HttpUrl(
        "https://agents-course-unit4-scoring.hf.space"
    )
    chess_eval_url: HttpUrl = HttpUrl(
        "https://stockfish.online/api/s/v2.php"
    )
    gemini_api_key: SecretStr
    langfuse_public_key: SecretStr
    langfuse_secret_key: SecretStr
    openrouter_api_key: SecretStr
    otel_exporter_otlp_endpoint: HttpUrl
    serper_api_key: SecretStr
    space_id: str
    username: str
    
    def set_langfuse_auth(self):
        LANGFUSE_AUTH = base64.b64encode(f"{self.langfuse_public_key.get_secret_value()}:{self.langfuse_secret_key.get_secret_value()}".encode()).decode()
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"