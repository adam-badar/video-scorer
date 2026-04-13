from pydantic import SecretStr
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    deepgram_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    supabase_url: str | None = None
    supabase_service_key: SecretStr | None = None
    api_key: SecretStr | None = None
    allowed_url_prefix: str = "https://fgdvuqqucvfxclfflffg.supabase.co/storage/v1/object/public/post-media/"

    model_config = {"env_prefix": "VIDEO_SCORER_"}


settings = Settings()
