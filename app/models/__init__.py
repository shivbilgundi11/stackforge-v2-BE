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
from app.models.billing import (
    BillingEvent,
    Metric,
    PlanQuota,
    Subscription,
    SubscriptionStatus,
    UsageRecord,
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
from app.models.organization import (
    Approval,
    ApprovalStatus,
    Comment,
    Invitation,
    Organization,
    OrganizationMember,
    OrgRole,
    TeamResourceType,
    Visibility,
)
from app.models.project import Project, ProjectItem, ProjectItemType
from app.models.stack import Stack, StackVersion
from app.models.template import Difficulty, Template, TemplateCategory
from app.models.tool_run import RunSource, ToolRun
from app.models.user import Plan, PlanSource, User, UserRole

__all__ = [
    "AiCall",
    "AiOutcome",
    "AnonymousSession",
    "Approval",
    "ApprovalStatus",
    "AuthEvent",
    "AuthEventType",
    "AuthOutcome",
    "AuthToken",
    "BillingEvent",
    "CatalogFlag",
    "Comment",
    "Compatibility",
    "DataSource",
    "Difficulty",
    "Export",
    "ExportFormat",
    "ExportStatus",
    "FlagStatus",
    "GpuPricing",
    "Invitation",
    "LifecycleStatus",
    "Metric",
    "ModelFamily",
    "ModelPricing",
    "OAuthAccount",
    "OAuthProvider",
    "OrgRole",
    "Organization",
    "OrganizationMember",
    "Plan",
    "PlanQuota",
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
    "Subscription",
    "SubscriptionStatus",
    "TeamResourceType",
    "Template",
    "TemplateCategory",
    "TokenPurpose",
    "Tool",
    "ToolCategory",
    "ToolRun",
    "ToolStatus",
    "UsageRecord",
    "User",
    "UserRole",
    "Visibility",
]
