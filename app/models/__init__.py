"""Every model must be imported here.

Alembic autogenerate walks `Base.metadata`, and a model that is never imported
is invisible to it — the failure mode is a migration that silently drops a
table nobody noticed was missing.
"""

from app.models.auth import (
    AnonymousSession,
    AuthEvent,
    AuthEventType,
    AuthOutcome,
    AuthToken,
    OAuthAccount,
    OAuthProvider,
    Session,
    TokenPurpose,
)
from app.models.user import Plan, PlanSource, User, UserRole

__all__ = [
    "AnonymousSession",
    "AuthEvent",
    "AuthEventType",
    "AuthOutcome",
    "AuthToken",
    "OAuthAccount",
    "OAuthProvider",
    "Plan",
    "PlanSource",
    "Session",
    "TokenPurpose",
    "User",
    "UserRole",
]
