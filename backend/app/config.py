from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MONGO_URI: str = "mongodb://localhost:27019"
    JWT_SECRET: str = "supersecretjwtkey_change_me_in_prod"
    JWT_EXPIRES_MIN: int = 60
    REDIS_URL: str = "redis://localhost:6381/0"
    MAX_UPLOAD_MB: int = 200
    MAX_VIDEO_SECONDS: int = 60

    # Model Integration Checkpoints (Provided by CrowdCore)
    TEXT_MODEL_PATH: str | None = None
    IMAGE_MODEL_PATH: str | None = None
    VIDEO_MODEL_PATH: str | None = None
    AUDIO_MODEL_PATH: str | None = None

    class Config:
        env_file = "../.env"
        extra = "ignore"

settings = Settings()
