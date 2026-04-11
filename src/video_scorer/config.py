from pydantic import SecretStr
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    deepgram_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    supabase_url: str | None = None
    supabase_anon_key: SecretStr | None = None

    model_config = {"env_prefix": "VIDEO_SCORER_"}


settings = Settings()
