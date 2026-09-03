"""The Stack Architect engine (M15).

Two layers, and the order matters:

    filter (hard constraints) → assemble by role → score → rank → synthesise

**Hard constraints eliminate. They do not penalise.** If sensitivity is
`regulated`, a managed US-only vector store is not a low-scoring option — it
is not an option. Softening a hard constraint into a score is how a
compliance-violating stack ends up ranked third instead of absent, and third
is a position a tired reader still picks.

Layer two (M16) chooses among the candidates this produces and writes the
prose. It never invents a component or a price: the model argues about options
the engine chose, which is what bounds how wrong a recommendation can be
(D-06).
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Final, NamedTuple

from app.schemas.catalog import CompatibilityOut, ToolOut
from app.schemas.tools import ToolWarning
from app.services import stack_score_service

#: Fit to recommend in a new design. `caution`, `deprecated`, and
#: `not_for_production` are excluded outright rather than ranked down — the
#: graveyard exists so people stop reaching for these, and putting one in a
#: fresh recommendation undoes that in a single screen.
RECOMMENDABLE: Final = frozenset({"recommended", "stable"})


class Role(NamedTuple):
    key: str
    label: str
    category: str
    #: A stack without this role is not a stack. Optional roles are dropped
    #: when nothing survives the constraints, rather than failing the request.
    required: bool
    why: str


#: Roles are defined from catalog categories that actually exist. A role for a
#: category the catalog does not carry would produce a permanently empty slot
#: and an unexplained gap on the result page.
ROLES: Final[tuple[Role, ...]] = (
    Role("llm", "LLM provider", "llm-provider", True, "Generation and reasoning."),
    Role("framework", "Framework", "rag-framework", True, "Retrieval and orchestration glue."),
    Role("vector_db", "Vector store", "vector-db", True, "Embedding storage and search."),
    Role("database", "Database", "database", True, "Application state."),
    Role("cache", "Cache", "cache", False, "Hot paths and rate limiting."),
    Role("orchestration", "Orchestration", "orchestration", False, "Durable background work."),
    Role("observability", "Observability", "observability", False, "Traces, evals, and cost."),
    Role("deployment", "Deployment", "deployment", True, "Where it runs."),
    # M25. Both optional, and that is the whole mechanism: `component_rows`
    # skips a role nothing filled, so a stack calling an API drops both rows
    # without a conditional anywhere in the renderer. Most of this audience
    # integrates a third-party API, and growing the common answer by two rows
    # to serve the minority who self-host would make it worse for everyone.
    Role("compute", "Compute", "gpu-cloud", False, "Where the weights run."),
    Role("guardrails", "Guardrails", "guardrails", False, "Input and output inspection."),
)

AGENT_USE_CASES: Final = frozenset({"agents", "automation", "coding"})

#: Under this, a stack has no room for a batch-oriented component in the
#: request path. The tags are the catalog's own.
REALTIME_LATENCY_MS: Final = 500
BATCH_TAGS: Final = frozenset({"batch", "dag", "workflows", "async-processing"})


class Requirements(NamedTuple):
    use_case: str
    scale_target: str
    monthly_budget: int
    team_skill: str
    latency_ms: int
    sensitivity: str
    deployment: str
    capabilities: tuple[str, ...]
    # M25. Defaulted to the answers that make both new roles disappear, so an
    # untouched form produces exactly the pre-M25 recommendation — which is
    # what keeps every saved stack and every M15 fixture valid.
    model_hosting: str = "api"
    workload: str = "inference"
    traffic: str = "steady"
    residency: str = "any"


#: The catalog's own word for a provider serving open weights. Used rather
#: than a new fact because three vendors already carry it.
OPEN_WEIGHTS_TAG: Final = "open-weights"

#: Fine-tuning and training need the machine for longer and at a different
#: shape; inference alone is the case the compute layer is optional for.
TRAINING_WORKLOADS: Final = frozenset({"fine-tuning", "training"})

#: What a fine-tune needs to hold on one card, in GB. Below this the vendor is
#: offering inference hardware, whatever it calls the tier.
FINE_TUNING_VRAM_GB: Final = 80

#: A training run wants a multi-GPU instance. Both figures are derived from the
#: GPU registry at seed time rather than typed per vendor, so a provider whose
#: fleet changes is one re-seed away from a correct answer.
TRAINING_GPU_COUNT: Final = 8

#: The two roles M25 adds. Neither belongs in every stack, so each is offered
#: only when the intake says it applies — see `_offers_role`.
OPTIONAL_M25_CATEGORIES: Final = frozenset({"gpu-cloud", "guardrails"})


class Elimination(NamedTuple):
    slug: str
    name: str
    constraint: str
    reason: str


def _offers_role(category: str, requirements: Requirements) -> bool:
    """Whether a stack shaped like this has the role at all (M25).

    Not an elimination, and deliberately not recorded as one. The exclusions
    table exists to explain why a tool the user *expected* is missing; a user
    who said they are calling an API did not expect six GPU vendors, and
    listing all six as "excluded" would bury the constraints that actually bit
    under noise the user already knew about.
    """
    if category == "gpu-cloud":
        # Only running the weights yourself puts a machine in the stack.
        # `managed-open-weights` is someone else's machine, and recommending a
        # GPU to a user who will never rent one is the layer bloat this module
        # exists to avoid.
        return requirements.model_hosting == "self-hosted"
    if category == "guardrails":
        # Sensitive data or an agent holding tools. Everything else gets the
        # eight roles it got before this module existed.
        return (
            requirements.sensitivity in {"confidential", "restricted", "regulated"}
            or requirements.use_case in AGENT_USE_CASES
        )
    return True


def _residency_reason(tool: ToolOut, region: str) -> str | None:
    """Why this region rules the tool out, or `None` if it does not.

    Empty `residency` is read two ways, and `self_hostable` is what separates
    them: software the user runs themselves is unconstrained, because it runs
    wherever they run it. A managed vendor with an empty array has no verified
    residency on file, and the answer to "we do not know" is exclusion rather
    than a silent pass — a stack that quietly kept a US-only vendor in an EU
    answer is the one failure this constraint exists to prevent.
    """
    if region == "any":
        return None
    if tool.self_hostable:
        return None
    if not tool.residency:
        return (
            f"{tool.name} is managed and the catalog carries no verified "
            f"residency for it, so it cannot be offered against a {region.upper()} "
            f"requirement."
        )
    if region not in tool.residency:
        return (
            f"{tool.name} is not operable in {region.upper()} — it is available in "
            f"{', '.join(sorted(code.upper() for code in tool.residency))}."
        )
    return None


def eliminate(
    catalog: list[ToolOut], requirements: Requirements
) -> tuple[list[ToolOut], list[Elimination]]:
    """Apply every hard constraint, and record what each one removed.

    The record is the point. A recommendation that silently dropped the tool
    the user was expecting reads as a broken engine; one that says "excluded
    because your data cannot leave your network" reads as a correct one.
    """
    survivors: list[ToolOut] = []
    removed: list[Elimination] = []

    restricted = requirements.sensitivity in {"restricted", "regulated"}
    realtime = requirements.latency_ms < REALTIME_LATENCY_MS
    beginner = requirements.team_skill == "beginner"

    for tool in catalog:
        facts = tool.facts or {}

        # M25's two roles, before anything else: a role this stack does not
        # have cannot have its components eliminated for a reason.
        if tool.category in OPTIONAL_M25_CATEGORIES and not _offers_role(
            tool.category, requirements
        ):
            continue

        if tool.status not in RECOMMENDABLE:
            removed.append(
                Elimination(
                    tool.slug,
                    tool.name,
                    "status",
                    tool.status_reason or f"Catalog status is {tool.status}.",
                )
            )
            continue

        if restricted and not tool.self_hostable:
            removed.append(
                Elimination(
                    tool.slug,
                    tool.name,
                    "data_sensitivity",
                    f"{requirements.sensitivity} data cannot leave your network, and "
                    f"{tool.name} cannot be self-hosted.",
                )
            )
            continue

        # `gpu-cloud` is exempt, and the exemption is the point of the
        # category: renting a machine and running vLLM on it *is* self-hosting.
        # `self_hostable` asks whether software can run on infrastructure the
        # user controls, which is not a question about a vendor who rents the
        # infrastructure. Sensitivity still applies below — someone else's
        # datacentre is still someone else's — and that is the constraint that
        # should remove a compute layer, not this one.
        if (
            requirements.deployment == "self-hosted"
            and not tool.self_hostable
            and tool.category != "gpu-cloud"
        ):
            removed.append(
                Elimination(
                    tool.slug, tool.name, "deployment", f"{tool.name} has no self-hosted option."
                )
            )
            continue

        if requirements.deployment == "managed" and not facts.get("managed", False):
            removed.append(
                Elimination(
                    tool.slug, tool.name, "deployment", f"{tool.name} has no managed offering."
                )
            )
            continue

        if beginner and float(facts.get("ops_burden", 3)) >= 4:
            removed.append(
                Elimination(
                    tool.slug,
                    tool.name,
                    "team_skill",
                    f"{tool.name} needs an operator. A team new to this will spend its "
                    f"time running it rather than building on it.",
                )
            )
            continue

        raw_floor = facts.get("min_monthly")
        floor = float(raw_floor) if raw_floor is not None else None
        over_budget = (
            floor is not None
            and requirements.monthly_budget > 0
            and floor > requirements.monthly_budget
        )
        if over_budget and floor is not None:
            removed.append(
                Elimination(
                    tool.slug,
                    tool.name,
                    "budget",
                    f"{tool.name} starts at ${floor:,.0f}/month, above the whole "
                    f"${requirements.monthly_budget:,} budget.",
                )
            )
            continue

        needed = stack_score_service.SCALE_TARGETS.get(requirements.scale_target, 2)
        if needed >= 5 and float(facts.get("scale_ceiling", 3)) < 4:
            removed.append(
                Elimination(
                    tool.slug,
                    tool.name,
                    "scale_target",
                    f"{tool.name} is not credible at the scale you described.",
                )
            )
            continue

        if realtime and BATCH_TAGS & set(tool.tags):
            removed.append(
                Elimination(
                    tool.slug,
                    tool.name,
                    "latency",
                    f"{tool.name} is a batch-oriented component; a "
                    f"{requirements.latency_ms}ms budget has no room for it in the "
                    f"request path.",
                )
            )
            continue

        # ── M25 ──────────────────────────────────────────────────────────────
        #
        # Hosting your own weights and calling someone's API are different
        # stacks, and the LLM row is where that shows. Without this a
        # self-hosted answer recommends a GPU cluster next to OpenAI, which
        # reads as an engine that did not understand the question.
        if tool.category == "llm-provider":
            if requirements.model_hosting == "self-hosted" and not tool.self_hostable:
                removed.append(
                    Elimination(
                        tool.slug,
                        tool.name,
                        "model_hosting",
                        f"{tool.name} is an API. You said the weights run on your own "
                        f"hardware, which needs a runtime you can deploy.",
                    )
                )
                continue

            # Both halves of the answer, not either: Ollama serves open
            # weights and Anthropic is someone else's machine, and neither one
            # on its own is what "open weights, run by someone else" asked for.
            if requirements.model_hosting == "managed-open-weights" and not (
                OPEN_WEIGHTS_TAG in tool.tags and facts.get("managed", False)
            ):
                removed.append(
                    Elimination(
                        tool.slug,
                        tool.name,
                        "model_hosting",
                        f"{tool.name} is not open weights served by someone else — "
                        f"which is the shape you asked for.",
                    )
                )
                continue

        # Residency is checked against every tool, not just the compute layer:
        # a vector store holding the embeddings is as much a data-residency
        # question as the machine serving the model.
        residency_reason = _residency_reason(tool, requirements.residency)
        if residency_reason is not None:
            removed.append(Elimination(tool.slug, tool.name, "residency", residency_reason))
            continue

        if tool.category == "gpu-cloud":
            vram = float(facts.get("max_vram_gb", 0))
            gpus = float(facts.get("max_gpu_count", 0))

            if requirements.workload in TRAINING_WORKLOADS and vram < FINE_TUNING_VRAM_GB:
                removed.append(
                    Elimination(
                        tool.slug,
                        tool.name,
                        "workload",
                        f"{tool.name} tops out at {vram:,.0f}GB on one card, and a "
                        f"{requirements.workload} run needs at least "
                        f"{FINE_TUNING_VRAM_GB}GB.",
                    )
                )
                continue

            if requirements.workload == "training" and gpus < TRAINING_GPU_COUNT:
                removed.append(
                    Elimination(
                        tool.slug,
                        tool.name,
                        "workload",
                        f"{tool.name}'s largest instance is {gpus:,.0f} GPUs. Training "
                        f"from scratch wants at least {TRAINING_GPU_COUNT} in one box.",
                    )
                )
                continue

            if requirements.traffic == "spiky" and not facts.get("scale_to_zero", False):
                removed.append(
                    Elimination(
                        tool.slug,
                        tool.name,
                        "traffic",
                        f"{tool.name} bills for a machine that is running, and spiky "
                        f"traffic spends most of the month paying for an idle GPU.",
                    )
                )
                continue

        survivors.append(tool)

    return survivors, removed


def _role_category(role: Role, requirements: Requirements) -> str:
    """Agent-shaped work wants an agent framework, not a RAG one."""
    if role.key == "framework" and requirements.use_case in AGENT_USE_CASES:
        return "agent-framework"
    return role.category


def assemble(
    survivors: list[ToolOut], requirements: Requirements, *, variants: int = 5
) -> list[list[ToolOut]]:
    """Build candidate stacks by role.

    Candidates are formed by walking each required role's ranked options in
    parallel — candidate *n* takes each role's *n*th-best surviving option.
    That produces genuinely different stacks rather than five near-identical
    ones differing in the cache, which is what a naive cartesian product
    truncated at five would give.
    """
    by_role: dict[str, list[ToolOut]] = {}
    for role in ROLES:
        category = _role_category(role, requirements)
        options = [tool for tool in survivors if tool.category == category]
        options.sort(key=lambda tool: tool.maturity_score, reverse=True)
        if options:
            by_role[role.key] = options

    if not by_role:
        return []

    depth = min(variants, max(len(options) for options in by_role.values()))
    candidates: list[list[ToolOut]] = []
    seen: set[tuple[str, ...]] = set()

    for index in range(depth):
        components: list[ToolOut] = []
        for role in ROLES:
            role_options = by_role.get(role.key)
            if not role_options:
                continue
            components.append(role_options[min(index, len(role_options) - 1)])

        signature = tuple(sorted(tool.slug for tool in components))
        if signature in seen:
            continue
        seen.add(signature)
        candidates.append(components)

    return candidates


class Candidate(NamedTuple):
    rank: int
    components: list[ToolOut]
    score: stack_score_service.StackScore
    compatibility: CompatibilityOut | None
    deprecated: list[ToolOut]


def rank(
    candidates: list[list[ToolOut]],
    requirements: Requirements,
    compatibilities: dict[int, CompatibilityOut],
    *,
    keep: int = 3,
) -> list[Candidate]:
    scored: list[Candidate] = []

    for index, components in enumerate(candidates):
        compatibility = compatibilities.get(index)
        stack_score = stack_score_service.score(
            components,
            monthly_budget=requirements.monthly_budget,
            scale_target=requirements.scale_target,
            sensitivity=requirements.sensitivity,
            compatibility=compatibility,
        )
        scored.append(
            Candidate(
                rank=0,
                components=components,
                score=stack_score,
                compatibility=compatibility,
                # Recomputed on every read, so a tool buried after this stack
                # was saved shows up the next time it is opened. FR-20.
                deprecated=[tool for tool in components if tool.status not in RECOMMENDABLE],
            )
        )

    scored.sort(key=lambda candidate: candidate.score.total, reverse=True)
    return [
        candidate._replace(rank=position) for position, candidate in enumerate(scored[:keep], 1)
    ]


def component_rows(
    candidate: Candidate,
    requirements: Requirements,
    *,
    approved: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """One row per role, with the graveyard status on the row itself.

    On the row rather than in a footnote: a stack that silently recommends a
    dead tool destroys the trust the whole product runs on, and a footnote is
    where warnings go to be unread (FR-20).

    When an organization's approved-tool list is in force, each row also says
    whether the tool is on it — the badge the UI shows next to unapproved
    picks (M21).
    """
    by_category = {tool.category: tool for tool in candidate.components}
    rows: list[dict[str, Any]] = []

    for role in ROLES:
        tool = by_category.get(_role_category(role, requirements))
        if tool is None:
            continue
        row = {
            "role": role.label,
            "role_key": role.key,
            "slug": tool.slug,
            "name": tool.name,
            "status": tool.status,
            "status_reason": tool.status_reason or "",
            "alternatives": ", ".join(tool.alternatives),
            "why": _why(tool, role, requirements),
            "self_hostable": "yes" if tool.self_hostable else "no",
            "pricing_model": tool.pricing_model or "—",
            "docs_url": tool.docs_url or "",
        }
        if approved:
            row["approved"] = "yes" if tool.slug in approved else "no"
        rows.append(row)
    return rows


def prefer_approved(
    ranked: list[Candidate], approved: frozenset[str], *, margin: int = 5
) -> list[Candidate]:
    """Approved-tool preference (M21).

    Among candidates within `margin` points of the leader, fewer unapproved
    components wins; the sort is stable, so score order survives inside each
    tier. A clearly better unapproved stack still wins — policy prefers,
    quality decides, and the flags say the rest.
    """
    if not approved or not ranked:
        return ranked

    def unapproved(candidate: Candidate) -> int:
        return sum(1 for tool in candidate.components if tool.slug not in approved)

    top = ranked[0].score.total
    near = [candidate for candidate in ranked if top - candidate.score.total <= margin]
    far = [candidate for candidate in ranked if top - candidate.score.total > margin]
    near.sort(key=unapproved)
    return [candidate._replace(rank=position) for position, candidate in enumerate(near + far, 1)]


def approved_flags(candidate: Candidate, approved: frozenset[str]) -> list[ToolWarning]:
    """Flag, never exclude (M21).

    A recommendation that quietly omits the best option because of a policy
    the user cannot see is worse than one that shows it with a badge and lets
    a human decide.
    """
    if not approved:
        return []
    return [
        ToolWarning(
            level="warning",
            message=(
                f"{tool.name} is not on your organization's approved tool list. "
                "It is shown because it best fits your requirements — confirm "
                "with your team before adopting it."
            ),
        )
        for tool in candidate.components
        if tool.slug not in approved
    ]


def _why(tool: ToolOut, role: Role, requirements: Requirements) -> str:
    """One sentence tying the choice to a stated requirement.

    Written from the constraint that selected it rather than from the tool's
    marketing: "highest-maturity option that runs inside your network" is
    checkable, "best-in-class vector database" is not.
    """
    facts = tool.facts or {}
    if requirements.sensitivity in {"restricted", "regulated"} and tool.self_hostable:
        return f"Runs inside your network, which {requirements.sensitivity} data requires."
    if requirements.team_skill == "beginner" and float(facts.get("ops_burden", 3)) <= 2:
        return "Lowest operational burden in its category — little to run or tune."
    if requirements.scale_target in {"large", "xlarge"}:
        return (
            f"Credible at your scale target, and the highest-maturity option available "
            f"for {role.label.lower()}."
        )
    return f"Highest-maturity {role.label.lower()} that satisfies every stated constraint."


def warnings_for(
    candidate: Candidate,
    requirements: Requirements,
    eliminations: list[Elimination],
) -> list[ToolWarning]:
    warnings: list[ToolWarning] = []

    for tool in candidate.deprecated:
        warnings.append(
            ToolWarning(
                level="critical",
                message=(
                    f"{tool.name} is marked {tool.status} in the catalog: "
                    f"{tool.status_reason or 'no reason recorded'}. "
                    + (f"Consider {', '.join(tool.alternatives)}." if tool.alternatives else "")
                ),
            )
        )

    if candidate.compatibility and candidate.compatibility.weakest_pair:
        pair = candidate.compatibility.weakest_pair
        if pair.score < 70:
            warnings.append(
                ToolWarning(
                    level="warning",
                    message=(
                        f"{pair.tool_a} and {pair.tool_b} score {pair.score}/100 together — "
                        f"the weakest pairing in this stack. " + (pair.notes or "")
                    ),
                )
            )

    if candidate.compatibility and candidate.compatibility.missing_pairs:
        warnings.append(
            ToolWarning(
                level="info",
                message=(
                    f"{len(candidate.compatibility.missing_pairs)} pairings in this stack have "
                    f"no reviewed compatibility score. They are reported as unknown rather "
                    f"than assumed to work."
                ),
            )
        )

    by_constraint: dict[str, int] = {}
    for elimination in eliminations:
        by_constraint[elimination.constraint] = by_constraint.get(elimination.constraint, 0) + 1
    for constraint, count in sorted(by_constraint.items()):
        if constraint == "status":
            continue
        warnings.append(
            ToolWarning(
                level="info",
                message=(
                    f"{count} tools were excluded by your {constraint.replace('_', ' ')} "
                    f"constraint before scoring. Hard constraints eliminate rather than "
                    f"rank down — see the exclusions table."
                ),
            )
        )

    if not candidate.components:
        warnings.append(
            ToolWarning(
                level="critical",
                message=(
                    "No stack satisfies every constraint at once. Relax the tightest one — "
                    "usually the budget or the sensitivity — and run it again."
                ),
            )
        )

    return warnings


def rule_summary(candidate: Candidate, requirements: Requirements) -> str:
    """What ships when synthesis is unavailable.

    Has to read as a real explanation rather than a placeholder: this is the
    text a user sees on the demo path when no key is configured.
    """
    names = [tool.name for tool in candidate.components]
    lead = names[0] if names else "no components"
    return (
        f"A {requirements.use_case} stack for {requirements.scale_target} scale on a "
        f"${requirements.monthly_budget:,}/month budget, scoring "
        f"{candidate.score.total}/100. Built around {lead}, with every component "
        f"satisfying your {requirements.sensitivity} sensitivity and "
        f"{requirements.deployment} deployment constraints — those eliminated options "
        f"rather than ranking them down, so nothing here is a compromise on them."
    )


def total_of(candidate: Candidate) -> Decimal:
    return candidate.score.total
