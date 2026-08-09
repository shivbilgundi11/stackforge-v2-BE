"""Typed settings.

One `Settings` object, cached. Missing required values fail at import — a
misconfigured deploy should refuse to boot rather than serve 500s.

Optional integrations expose an `*_enabled` property derived from key presence.
`ai_enabled` is what lets the AI layer degrade cleanly instead of scattering
`if key is None` through the services.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

Environment = Literal["local", "preview", "production"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ────────────────────────────────────────────────────────────────
    environment: Environment = "local"
    debug: bool = False
    api_base_url: str = "http://localhost:8000"
    web_base_url: str = "http://localhost:3000"
    # NoDecode: pydantic-settings JSON-parses complex types straight from the
    # environment, which fails on a comma-separated list before any validator
    # runs. This hands the raw string to `_split_origins` instead.
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:3000"]
    )

    # ── Database ───────────────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://stackforge:stackforge@localhost:5432/stackforge"
    database_pool_size: int = 10
    database_max_overflow: int = 20
    database_echo: bool = False

    # ── Redis ──────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Auth ───────────────────────────────────────────────────────────────
    auth_private_key: str = ""
    auth_public_key: str = ""
    auth_key_id: str = "k1"

    access_token_ttl_seconds: int = 900
    refresh_token_ttl_days: int = 30
    refresh_token_absolute_ttl_days: int = 90
    email_verify_ttl_hours: int = 24
    password_reset_ttl_minutes: int = 30

    refresh_cookie_name: str = "__sf_rt"
    anon_cookie_name: str = "__sf_anon"
    cookie_domain: str | None = None
    cookie_secure: bool = False

    argon2_time_cost: int = 3
    argon2_memory_cost_kib: int = 65536
    argon2_parallelism: int = 4

    max_failed_logins: int = 5
    lockout_seconds: int = 900
    hibp_enabled: bool = True

    # ── Email ──────────────────────────────────────────────────────────────
    email_provider: Literal["smtp", "resend", "console"] = "console"
    email_from: str = "StackForge <noreply@localhost>"
    smtp_host: str = "localhost"
    smtp_port: int = 1025
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_tls: bool = False
    resend_api_key: str = ""

    # ── OAuth ──────────────────────────────────────────────────────────────
    google_client_id: str = ""
    google_client_secret: str = ""
    github_client_id: str = ""
    github_client_secret: str = ""

    # ── AI ─────────────────────────────────────────────────────────────────
    anthropic_api_key: str = ""

    # ── Observability ──────────────────────────────────────────────────────
    sentry_dsn: str = ""
    log_level: str = "INFO"
    log_json: bool = False

    # ───────────────────────────────────────────────────────────────────────

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("auth_private_key", "auth_public_key", mode="before")
    @classmethod
    def _unescape_pem(cls, value: object) -> object:
        # PEM keys live in .env as one line with literal \n. dotenv does not
        # unescape those, and cryptography rejects a key without real newlines.
        if isinstance(value, str):
            return value.replace("\\n", "\n").strip()
        return value

    @field_validator("cookie_domain", mode="before")
    @classmethod
    def _blank_domain_is_none(cls, value: object) -> object:
        # A blank COOKIE_DOMAIN must mean host-only, not the empty string — an
        # empty Domain attribute is not valid in a Set-Cookie.
        #
        # The `#` guard is not paranoia: dotenv keeps an inline comment as the
        # value when the value itself is blank, which silently produced
        # `Domain=# blank = host-only` on every cookie and broke the whole
        # session flow.
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned or cleaned.startswith("#"):
                return None
            return cleaned
        return value or None

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def ai_enabled(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def google_oauth_enabled(self) -> bool:
        return bool(self.google_client_id and self.google_client_secret)

    @property
    def github_oauth_enabled(self) -> bool:
        return bool(self.github_client_id and self.github_client_secret)

    def validate_runtime(self) -> None:
        """Checks that must hold before the app serves traffic.

        Called from the lifespan rather than a validator so tests can build a
        Settings object without a signing key.
        """
        problems: list[str] = []

        if not self.auth_private_key or not self.auth_public_key:
            problems.append(
                "AUTH_PRIVATE_KEY and AUTH_PUBLIC_KEY are required. "
                "Generate them with: uv run python -m app.cli generate-keypair"
            )

        if self.is_production:
            if not self.cookie_secure:
                problems.append("COOKIE_SECURE must be true in production.")
            if self.debug:
                problems.append("DEBUG must be false in production.")
            if "*" in self.cors_origins:
                problems.append("CORS_ORIGINS must not contain '*' in production.")
            if self.email_provider == "console":
                problems.append("EMAIL_PROVIDER=console will not deliver mail in production.")

        if problems:
            raise RuntimeError("Invalid configuration:\n  - " + "\n  - ".join(problems))


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
