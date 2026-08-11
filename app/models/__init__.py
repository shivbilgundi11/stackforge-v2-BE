"""Every model must be imported here.

Alembic autogenerate walks `Base.metadata`, and a model that is never imported
is invisible to it — the failure mode is a migration that silently drops a
table nobody noticed was missing.
"""

from app.models.ai import AiCall, AiOutcome
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
from app.models.catalog import (
    CatalogFlag,
    Compatibility,
    DataSource,
    FlagStatus,
    GpuPricing,
    LifecycleStatus,
    ModelFamily,
    ModelPricing,
    PricedEntity,
    PricingHistory,
    SourceKind,
    Tool,
    ToolCategory,
    ToolStatus,
)
from app.models.export import (
    Export,
    ExportFormat,
    ExportStatus,
    ShareLink,
    SourceType,
)
from app.models.project import Project, ProjectItem, ProjectItemType
from app.models.stack import Stack, StackVersion
from app.models.tool_run import RunSource, ToolRun
from app.models.user import Plan, PlanSource, User, UserRole

__all__ = [
    "AiCall",
    "AiOutcome",
    "AnonymousSession",
    "AuthEvent",
    "AuthEventType",
    "AuthOutcome",
    "AuthToken",
    "CatalogFlag",
    "Compatibility",
    "DataSource",
    "Export",
    "ExportFormat",
    "ExportStatus",
    "FlagStatus",
    "GpuPricing",
    "LifecycleStatus",
    "ModelFamily",
    "ModelPricing",
    "OAuthAccount",
    "OAuthProvider",
    "Plan",
    "PlanSource",
    "PricedEntity",
    "PricingHistory",
    "Project",
    "ProjectItem",
    "ProjectItemType",
    "RunSource",
    "Session",
    "ShareLink",
    "SourceKind",
    "SourceType",
    "Stack",
    "StackVersion",
    "TokenPurpose",
    "Tool",
    "ToolCategory",
    "ToolRun",
    "ToolStatus",
    "User",
    "UserRole",
]
