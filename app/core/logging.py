"""Structured logging.

JSON everywhere except local, where it is human-readable. Every line inside a
request carries the request id and identity, injected from the context var.

Secrets never reach the output. The redaction processor is enforced by a test,
not by discipline.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from app.core.config import settings
from app.core.context import as_dict

REDACTED = "[redacted]"

# Substring match, lowercased. Deliberately broad — a false positive costs a
# hidden log value, a false negative costs a leaked credential.
SENSITIVE_KEYS = (
    "password",
    "passwd",
    "secret",
    "token",
    "authorization",
    "cookie",
    "api_key",
    "apikey",
    "private_key",
    "client_secret",
    "credential",
    "session",
)


def _redact(value: Any, depth: int = 0) -> Any:
    if depth > 6:
        return value
    if isinstance(value, dict):
        return {
            key: (
                REDACTED
                if any(marker in str(key).lower() for marker in SENSITIVE_KEYS)
                else _redact(item, depth + 1)
            )
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [_redact(item, depth + 1) for item in value]
    return value


def redact_processor(
    _logger: Any, _name: str, event_dict: structlog.types.EventDict
) -> structlog.types.EventDict:
    result = _redact(dict(event_dict))
    assert isinstance(result, dict)
    return result


def context_processor(
    _logger: Any, _name: str, event_dict: structlog.types.EventDict
) -> structlog.types.EventDict:
    for key, value in as_dict().items():
        event_dict.setdefault(key, value)
    return event_dict


def configure_logging() -> None:
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
    )

    # Uvicorn's own access log duplicates our request middleware and is not
    # structured. Ours carries more and is correlated.
    logging.getLogger("uvicorn.access").disabled = True

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if settings.log_json or settings.is_production
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            context_processor,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            redact_processor,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "stackforge") -> structlog.stdlib.BoundLogger:
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger
