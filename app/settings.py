"""Service configuration. Every value comes from the environment (prefix KSL_); see .env.example."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KSL_", env_file=".env", extra="ignore")

    env: str = "dev"
    log_level: str = "INFO"
    public_base_url: str = "http://localhost:8000"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/ksl"
    database_url_sync: str = "postgresql://postgres:postgres@localhost:5432/ksl"
    redis_url: str = "redis://localhost:6379/0"

    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "minio"
    s3_secret_key: str = Field(default="", repr=False)
    s3_region: str = "us-east-1"
    s3_bucket_datasets: str = "datasets"
    s3_bucket_models: str = "models"
    s3_bucket_kb: str = "kb"
    s3_bucket_sign_dict: str = "sign-dict"

    jwt_kid: str = "dev"
    jwt_private_key_pem: str = Field(default="", repr=False)
    jwt_public_key_pem: str = ""
    jwt_issuer: str = "ksl-bridge"
    jwt_audience: str = "ksl-bridge-clients"
    access_ttl_seconds: int = 900
    refresh_ttl_days: int = 30
    guest_ttl_seconds: int = 7200
    api_key_pepper: str = Field(default="dev-pepper-change-me", repr=False)
    otp_pepper: str = Field(default="dev-otp-pepper", repr=False)
    data_encryption_key: str = Field(default="", repr=False)

    sms_provider: str = "console"
    sms_api_key: str = Field(default="", repr=False)
    sms_sender_id: str = "KSLBRIDGE"

    ws_max_frame_bytes: int = 65536
    ws_heartbeat_seconds: int = 15
    ws_idle_timeout_seconds: int = 60
    ws_max_inflight_landmark_batches: int = 8

    endpoint_still_ms: int = 450
    max_utterance_seconds: int = 20
    clarify_cost: float = 0.15
    job_deadline_ms: int = 1500

    metrics_enabled: bool = True
    otel_exporter_otlp_endpoint: str = ""

    @property
    def is_production(self) -> bool:
        return self.env in ("staging", "prod")


@lru_cache
def get_settings() -> Settings:
    return Settings()
