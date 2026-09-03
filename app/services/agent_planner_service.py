"""Multi-agent topology rules (WF3).

Rules first, AI second (D-06). The engine picks the topology, assigns roles,
writes the handoff contracts, prices every node, and enumerates the failure
modes. Only that finished structure is handed to a model to write prose about,
so with no AI available the tool returns the same topology, the same diagram,
and the same costs with a rule-written summary marked `rule_based` — a degraded
explanation of a correct answer, never a degraded answer.

The failure modes are the part worth reading. A multi-agent system fails in
ways specific to its shape: a parallel fan-out duplicates work and produces
contradictory findings, a hierarchy overflows the supervisor's context, a
handoff chain loses the thread of what the user originally asked. Naming them
against the topology that was actually chosen is more use than a generic list
of agent risks, because the mitigation differs for each.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Final, NamedTuple

from app.schemas.catalog import ModelOut, ToolOut
from app.schemas.tools import Artifact, ToolOutput, ToolWarning
from app.services import diagram_theme
from app.services.cost_service import DAYS_PER_MONTH, THOUSAND

CENTS: Final = Decimal("0.01")
MICRO: Final = Decimal("0.000001")

STYLES: Final = ("sequential", "parallel", "hierarchical", "handoff")

#: Fit to recommend in a new design, matching WF2. `caution` entries are
#: excluded rather than ranked down — which for this workflow removes CrewAI
#: and AutoGen, the two frameworks people reach for first when they hear
#: "hierarchical". That is worth a warning rather than a silent omission.
RECOMMENDABLE: Final = frozenset({"recommended", "stable"})

#: Per-step token profile by role, and how many steps that role runs per task.
#: Defaults, not measurements — they exist so the plan carries a cost at all.
#: The agent cost calculator is where a real profile gets entered, which is
#: what the handoff from this tool is for.
PROFILES: Final[dict[str, tuple[int, int, int]]] = {
    # role: (input tokens/step, output tokens/step, steps per task)
    "planner": (1_500, 600, 1),
    "supervisor": (2_500, 500, 3),
    "lead": (1_800, 500, 2),
    "worker": (1_200, 400, 3),
    "aggregator": (2_500, 700, 1),
    "reviewer": (2_000, 400, 1),
    "triage": (800, 200, 1),
    "specialist": (1_500, 500, 2),
    "resolver": (1_800, 400, 1),
}

#: Which roles need the reasoning and which need throughput. Routing every node
#: to a frontier model is the most common reason an agent design is three times
#: more expensive than it needs to be.
FRONTIER_ROLES: Final = frozenset({"planner", "supervisor", "aggregator", "reviewer", "resolver"})

FRAMEWORK_PREFERENCE: Final[dict[str, tuple[str, ...]]] = {
    "sequential": ("pydantic-ai", "langgraph", "claude-agent-sdk"),
    "parallel": ("langgraph", "claude-agent-sdk", "pydantic-ai"),
    "hierarchical": ("langgraph", "claude-agent-sdk"),
    "handoff": ("openai-agents-sdk", "langgraph", "claude-agent-sdk"),
}

#: Above this many nodes, or with any node that can loop, the workflow needs
#: durable execution rather than a process that holds state in memory. A
#: six-agent run that dies at agent five and restarts from zero is the failure
#: that turns a working demo into an unshippable one.
DURABILITY_THRESHOLD: Final = 5


def _money(value: Decimal) -> Decimal:
    return value.quantize(MICRO, rounding=ROUND_HALF_UP)


def _display(value: Decimal) -> Decimal:
    return value.quantize(CENTS, rounding=ROUND_HALF_UP)


def _usd(value: Decimal) -> str:
    return f"${value:,.2f}" if value >= 1 else f"${value:,.6f}".rstrip("0").rstrip(".")


class Node(NamedTuple):
    id: str
    role: str
    title: str
    responsibility: str
    tools: tuple[str, ...]


class Edge(NamedTuple):
    source: str
    target: str
    contract: str


# ── topology ─────────────────────────────────────────────────────────────────


def _chunk(tools: list[str], groups: int) -> list[tuple[str, ...]]:
    """Split the tool list into `groups` roughly equal buckets, none empty."""
    if not tools:
        return [() for _ in range(groups)]
    groups = max(1, min(groups, len(tools)))
    size, extra = divmod(len(tools), groups)
    buckets: list[tuple[str, ...]] = []
    start = 0
    for index in range(groups):
        end = start + size + (1 if index < extra else 0)
        buckets.append(tuple(tools[start:end]))
        start = end
    return buckets


def topology(style: str, tools: list[str]) -> tuple[list[Node], list[Edge]]:
    """Nodes and edges for a coordination style.

    Worker count follows the tool list rather than being asked for: an agent
    exists to own a set of capabilities, and a roster chosen independently of
    the tools produces either agents with nothing to do or agents carrying
    every tool — which is the schema-overhead problem the cost calculator
    exists to show.
    """
    worker_count = max(1, min(4, (len(tools) + 2) // 3)) if tools else 2

    match style:
        case "parallel":
            return _parallel(tools, worker_count)
        case "hierarchical":
            return _hierarchical(tools, worker_count)
        case "handoff":
            return _handoff(tools, worker_count)
        case _:
            return _sequential(tools, worker_count)


def _sequential(tools: list[str], worker_count: int) -> tuple[list[Node], list[Edge]]:
    buckets = _chunk(tools, worker_count)
    nodes = [
        Node(
            "planner",
            "planner",
            "Planner",
            "Turns the goal into an ordered list of concrete steps and stops when the goal is met.",
            (),
        )
    ]
    edges: list[Edge] = []
    previous = "planner"

    for index, bucket in enumerate(buckets, start=1):
        node_id = f"step_{index}"
        nodes.append(
            Node(
                node_id,
                "worker",
                f"Step {index}",
                f"Executes step {index} using {', '.join(bucket) or 'no external tools'}, and "
                f"returns a result plus whether it succeeded.",
                bucket,
            )
        )
        edges.append(
            Edge(
                previous,
                node_id,
                "Step definition and the accumulated result so far. Receiver must return "
                "`{result, succeeded, error?}` — never a bare string.",
            )
        )
        previous = node_id

    nodes.append(
        Node(
            "reviewer",
            "reviewer",
            "Reviewer",
            "Checks the final output against the original goal and rejects once, at most.",
            (),
        )
    )
    edges.append(
        Edge(
            previous,
            "reviewer",
            "Final output plus the full step log. The reviewer sees what was attempted, "
            "not only what came out.",
        )
    )
    return nodes, edges


def _parallel(tools: list[str], worker_count: int) -> tuple[list[Node], list[Edge]]:
    buckets = _chunk(tools, max(2, worker_count))
    nodes = [
        Node(
            "dispatcher",
            "planner",
            "Dispatcher",
            "Splits the goal into independent units of work and fans them out. Units must "
            "not depend on each other's output.",
            (),
        )
    ]
    edges: list[Edge] = []

    for index, bucket in enumerate(buckets, start=1):
        node_id = f"worker_{index}"
        nodes.append(
            Node(
                node_id,
                "worker",
                f"Worker {index}",
                f"Owns {', '.join(bucket) or 'a share of the work'} and returns a finding "
                f"with the evidence behind it.",
                bucket,
            )
        )
        edges.append(
            Edge(
                "dispatcher",
                node_id,
                "One self-contained unit of work with everything needed to complete it. No "
                "shared mutable state.",
            )
        )
        edges.append(
            Edge(
                node_id,
                "aggregator",
                "`{finding, evidence, confidence}`. Confidence is required — the aggregator "
                "cannot resolve a disagreement between two unqualified assertions.",
            )
        )

    nodes.append(
        Node(
            "aggregator",
            "aggregator",
            "Aggregator",
            "Merges the findings, resolves contradictions explicitly, and reports which "
            "workers disagreed.",
            (),
        )
    )
    return nodes, edges


def _hierarchical(tools: list[str], worker_count: int) -> tuple[list[Node], list[Edge]]:
    lead_count = 2 if worker_count > 2 else 1
    buckets = _chunk(tools, max(lead_count * 2, 2))
    nodes = [
        Node(
            "supervisor",
            "supervisor",
            "Supervisor",
            "Owns the goal, delegates to leads, and decides when the work is done. Never "
            "calls a tool itself.",
            (),
        )
    ]
    edges: list[Edge] = []
    bucket_index = 0

    for lead_index in range(1, lead_count + 1):
        lead_id = f"lead_{lead_index}"
        nodes.append(
            Node(
                lead_id,
                "lead",
                f"Lead {lead_index}",
                "Breaks its assignment into worker tasks and reports one consolidated "
                "result upward.",
                (),
            )
        )
        edges.append(
            Edge(
                "supervisor",
                lead_id,
                "An objective and a budget — step count and spend. A delegation with no "
                "budget is how a hierarchy runs away.",
            )
        )
        for _ in range(2):
            if bucket_index >= len(buckets):
                break
            worker_id = f"worker_{bucket_index + 1}"
            bucket = buckets[bucket_index]
            nodes.append(
                Node(
                    worker_id,
                    "worker",
                    f"Worker {bucket_index + 1}",
                    f"Executes with {', '.join(bucket) or 'its assigned tools'} and reports "
                    f"success or failure with a reason.",
                    bucket,
                )
            )
            edges.append(
                Edge(
                    lead_id,
                    worker_id,
                    "A single task and its acceptance criteria. The worker never sees the "
                    "overall goal, which keeps its context small.",
                )
            )
            bucket_index += 1

    return nodes, edges


def _handoff(tools: list[str], worker_count: int) -> tuple[list[Node], list[Edge]]:
    buckets = _chunk(tools, max(2, worker_count))
    nodes = [
        Node(
            "triage",
            "triage",
            "Triage",
            "Classifies the request and hands to exactly one specialist. Cheap, fast, and "
            "does no work itself.",
            (),
        )
    ]
    edges: list[Edge] = []

    for index, bucket in enumerate(buckets, start=1):
        node_id = f"specialist_{index}"
        nodes.append(
            Node(
                node_id,
                "specialist",
                f"Specialist {index}",
                f"Handles its category end to end with {', '.join(bucket) or 'its own tools'}, "
                f"or hands back if it was the wrong choice.",
                bucket,
            )
        )
        edges.append(
            Edge(
                "triage",
                node_id,
                "The original request verbatim plus the classification and why. Passing a "
                "summary instead is how the user's actual words get lost.",
            )
        )
        edges.append(
            Edge(
                node_id,
                "resolver",
                "The answer, plus every handoff it went through. The trail is what makes a "
                "wrong routing debuggable.",
            )
        )

    nodes.append(
        Node(
            "resolver",
            "resolver",
            "Resolver",
            "Confirms the request was answered rather than merely routed, and closes it.",
            (),
        )
    )
    return nodes, edges


# ── failure modes ────────────────────────────────────────────────────────────

GENERIC_FAILURES: Final[tuple[tuple[str, str, str], ...]] = (
    (
        "Loop without termination",
        "high",
        "Cap total steps per task and per agent, and fail loudly at the cap. A step budget "
        "is the only reliable stop condition — a model asked to decide when it is finished "
        "will keep going.",
    ),
    (
        "Tool failure cascade",
        "high",
        "Treat a tool error as data the agent receives, not an exception that kills the run. "
        "Retry the tool twice, then hand the failure to the agent to route around.",
    ),
    (
        "Cost runaway on a single task",
        "medium",
        "Enforce a per-task spend ceiling in the orchestrator, not in the prompt. A prompt "
        "asking the model to be economical is not a control.",
    ),
)

STYLE_FAILURES: Final[dict[str, tuple[tuple[str, str, str], ...]]] = {
    "sequential": (
        (
            "Error propagation down the chain",
            "high",
            "Every step returns success explicitly. A step that silently passes a partial "
            "result forward produces an answer built on it, and the failure surfaces at the "
            "end where it cannot be attributed.",
        ),
        (
            "Latency is the sum of every step",
            "medium",
            "Nothing overlaps here. If the total exceeds the user's patience, the fix is a "
            "different topology, not a faster model.",
        ),
    ),
    "parallel": (
        (
            "Duplicated work across workers",
            "high",
            "Partition the work in the dispatcher and make units disjoint by construction. "
            "Workers cannot coordinate — they cannot see each other.",
        ),
        (
            "Contradictory findings with no resolution",
            "high",
            "Require a confidence and the evidence with every finding, and give the "
            "aggregator an explicit rule for ties. Without one it silently picks the "
            "longest answer.",
        ),
        (
            "Fan-out cost multiplies without a matching gain",
            "medium",
            "Price the fan-out before building it. Five workers cost five times one and are "
            "worth it only when the units are genuinely independent.",
        ),
    ),
    "hierarchical": (
        (
            "Supervisor context overflow",
            "high",
            "The supervisor accumulates every report. Have leads return summaries with a "
            "hard token ceiling, and keep the full detail in storage the supervisor can "
            "query rather than in its context.",
        ),
        (
            "Delegation without a budget",
            "high",
            "Every delegation carries a step and spend budget. Without one, a lead that "
            "cannot finish keeps trying and the supervisor cannot tell it apart from a lead "
            "doing careful work.",
        ),
        (
            "Responsibility diffusion",
            "medium",
            "Exactly one node owns the final answer. When both the supervisor and a lead "
            "believe the other is checking, nothing is checked.",
        ),
    ),
    "handoff": (
        (
            "The original request is lost at handoff",
            "high",
            "Pass the user's words verbatim, not a summary. Each rewrite loses detail, and "
            "by the third specialist the answer is to a question nobody asked.",
        ),
        (
            "Handoff ping-pong",
            "high",
            "Cap handoffs per request — two is usually right — and escalate to a human at "
            "the cap. Two specialists that each believe the other owns it will pass forever.",
        ),
        (
            "Misrouting at triage looks like a bad answer",
            "medium",
            "Log the classification and the confidence. A specialist answering well from "
            "the wrong category is indistinguishable from a weak model until you can see "
            "the routing.",
        ),
    ),
}


# ── the plan ─────────────────────────────────────────────────────────────────


def plan(
    *,
    goal: str,
    coordination: str,
    available_tools: list[str],
    constraints: list[str],
    tasks_per_day: int,
    frontier_model: ModelOut,
    fast_model: ModelOut | None,
    catalog: list[ToolOut],
) -> ToolOutput:
    """A costed topology with contracts, failure modes, and a framework."""
    nodes, edges = topology(coordination, available_tools)
    fast = fast_model or frontier_model

    node_rows: list[dict[str, Any]] = []
    per_task = Decimal(0)
    sourced_from: list[str] = [frontier_model.id]
    if fast_model is not None:
        sourced_from.append(fast_model.id)

    for node in nodes:
        input_tokens, output_tokens, steps = PROFILES[node.role]
        model = frontier_model if node.role in FRONTIER_ROLES else fast
        cost = Decimal(input_tokens * steps) / THOUSAND * model.input_cost_per_1k + Decimal(
            output_tokens * steps
        ) / THOUSAND * (model.output_cost_per_1k or Decimal(0))
        per_task += cost
        node_rows.append(
            {
                "node": node.title,
                "role": node.role,
                "model": model.display_name,
                "steps": steps,
                "tools": ", ".join(node.tools) or "—",
                "cost_per_task": _usd(_money(cost)),
                "responsibility": node.responsibility,
            }
        )

    per_day = per_task * Decimal(tasks_per_day)
    per_month = per_day * DAYS_PER_MONTH

    framework, framework_note = _pick_framework(catalog, coordination)
    durable = _pick_durable(catalog) if len(nodes) >= DURABILITY_THRESHOLD else None

    failures = [
        {"mode": mode, "likelihood": likelihood, "mitigation": mitigation}
        for mode, likelihood, mitigation in (
            *STYLE_FAILURES.get(coordination, ()),
            *GENERIC_FAILURES,
        )
    ]

    contracts = [
        {
            "from": _title_of(nodes, edge.source),
            "to": _title_of(nodes, edge.target),
            "contract": edge.contract,
        }
        for edge in edges
    ]

    diagram = _mermaid(nodes, edges)
    summary = _summary(
        goal=goal,
        coordination=coordination,
        nodes=nodes,
        framework=framework,
        per_task=per_task,
    )

    warnings = _plan_warnings(
        coordination=coordination,
        nodes=nodes,
        catalog=catalog,
        framework=framework,
        framework_note=framework_note,
        durable=durable,
        constraints=constraints,
        fast_model=fast_model,
    )

    if framework is not None:
        sourced_from.append(framework.slug)

    return ToolOutput(
        metrics={
            "topology": coordination,
            "agents": len(nodes),
            "handoffs": len(edges),
            "cost_per_task": _money(per_task),
            "cost_per_day": _display(per_day),
            "cost_per_month": _display(per_month),
            "framework": framework.name if framework else "none recommendable",
            "summary": summary,
        },
        tables={"nodes": node_rows, "contracts": contracts, "failure_modes": failures},
        artifacts=[
            Artifact(
                type="diagram",
                format="mermaid",
                filename="agent-topology.mmd",
                content=diagram,
            ),
            Artifact(
                type="plan",
                format="markdown",
                filename="agent-plan.md",
                content=_document(
                    goal=goal,
                    coordination=coordination,
                    summary=summary,
                    diagram=diagram,
                    node_rows=node_rows,
                    contracts=contracts,
                    failures=failures,
                    framework=framework,
                    per_task=per_task,
                    per_month=per_month,
                    constraints=constraints,
                ),
            ),
        ],
        warnings=warnings,
        sourced_from=sourced_from,
    )


def _title_of(nodes: list[Node], node_id: str) -> str:
    return next((node.title for node in nodes if node.id == node_id), node_id)


def _pick_framework(catalog: list[ToolOut], coordination: str) -> tuple[ToolOut | None, str]:
    """Highest-maturity recommendable framework, preferring ones fit for the shape."""
    frameworks = [
        tool
        for tool in catalog
        if tool.category == "agent-framework" and tool.status in RECOMMENDABLE
    ]
    excluded = [
        tool.name
        for tool in catalog
        if tool.category == "agent-framework" and tool.status not in RECOMMENDABLE
    ]

    for slug in FRAMEWORK_PREFERENCE.get(coordination, ()):
        for tool in frameworks:
            if tool.slug == slug:
                return tool, ", ".join(excluded)

    frameworks.sort(key=lambda tool: tool.maturity_score, reverse=True)
    return (frameworks[0] if frameworks else None), ", ".join(excluded)


def _pick_durable(catalog: list[ToolOut]) -> ToolOut | None:
    candidates = [
        tool
        for tool in catalog
        if tool.category == "orchestration" and tool.status in RECOMMENDABLE
    ]
    candidates.sort(key=lambda tool: tool.maturity_score, reverse=True)
    return candidates[0] if candidates else None


def _mermaid(nodes: list[Node], edges: list[Edge]) -> str:
    """Top-down DAG.

    Node ids are generated by this module from a fixed vocabulary, so nothing
    user-supplied reaches an identifier position. Labels are quoted and have
    quotes and newlines stripped — a tool name with a bracket in it otherwise
    produces a diagram that silently fails to parse in the renderer.
    """
    lines = ["graph TD"]
    roles: dict[str, str] = {}
    for node in nodes:
        label = _label(node.title)
        role = _label(node.role)
        lines.append(f'    {node.id}["{label}<br/><small>{role}</small>"]')
        roles[node.id] = node.role
    for edge in edges:
        lines.append(f"    {edge.source} --> {edge.target}")

    # Coloured, but never branded: a node here is a unit of work this module
    # invented, not a catalog entry, so there is nothing to look a logo up by.
    # The colour is what separates the parts that decide from the parts that
    # do, which is the question a reader brings to a topology.
    return "\n".join(diagram_theme.decorate(lines, roles=roles, tools=dict.fromkeys(roles)))


def _label(text: str) -> str:
    return str(text).replace('"', "").replace("\n", " ").replace("[", "(").replace("]", ")")


def _summary(
    *,
    goal: str,
    coordination: str,
    nodes: list[Node],
    framework: ToolOut | None,
    per_task: Decimal,
) -> str:
    """The rule-written rationale, and what ships when AI is unavailable.

    M16 replaces this with a synthesised version. It does not replace the
    topology, the contracts, or the costs, which are already correct.
    """
    workers = sum(1 for node in nodes if node.role in {"worker", "specialist", "lead"})
    shape = {
        "sequential": "a chain, each step gated on the one before it",
        "parallel": "a fan-out into independent workers with one aggregator",
        "hierarchical": "a supervisor delegating through leads",
        "handoff": "a triage step routing to one specialist at a time",
    }[coordination]

    parts = [
        f"{len(nodes)} agents arranged as {shape}, for: {goal.strip()[:200]}",
        f"{workers} of them do the work; the rest coordinate, and coordination is not free — "
        f"at {_usd(_money(per_task))} per task it is the overhead you are buying "
        f"reliability with.",
    ]
    if framework is not None:
        parts.append(
            f"{framework.name} is the recommended framework for this shape: {framework.description}"
        )
    return " ".join(parts)


def _plan_warnings(
    *,
    coordination: str,
    nodes: list[Node],
    catalog: list[ToolOut],
    framework: ToolOut | None,
    framework_note: str,
    durable: ToolOut | None,
    constraints: list[str],
    fast_model: ModelOut | None,
) -> list[ToolWarning]:
    warnings: list[ToolWarning] = []

    if len(nodes) <= 2:
        warnings.append(
            ToolWarning(
                level="info",
                message=(
                    "Two agents is barely a topology. If one agent with these tools can do "
                    "the job, it will be cheaper, faster, and far easier to debug — "
                    "multi-agent earns its cost when roles genuinely differ."
                ),
            )
        )
    if len(nodes) >= 6:
        warnings.append(
            ToolWarning(
                level="warning",
                message=(
                    f"{len(nodes)} agents is a distributed system. Every edge is a place "
                    f"context is lost and a place the run can stall, and debugging one is "
                    f"materially harder than debugging a chain of three."
                ),
            )
        )

    if coordination == "hierarchical" and framework_note:
        warnings.append(
            ToolWarning(
                level="info",
                message=(
                    f"{framework_note} are the usual first answer for a hierarchy and are "
                    f"excluded here — the catalog marks them `caution`. Check the graveyard "
                    f"entry before overriding that."
                ),
            )
        )

    if framework is None:
        warnings.append(
            ToolWarning(
                level="warning",
                message=(
                    "No framework in the catalog is currently recommendable for this shape. "
                    "The topology stands on its own — it is implementable directly against "
                    "a provider SDK."
                ),
            )
        )

    if durable is not None:
        warnings.append(
            ToolWarning(
                level="warning",
                message=(
                    f"At {len(nodes)} agents this is long-running work. A crash at the last "
                    f"node restarts from zero unless state is durable — {durable.name} or "
                    f"equivalent belongs under this, not a for-loop in a request handler."
                ),
            )
        )

    if fast_model is None:
        warnings.append(
            ToolWarning(
                level="info",
                field="fast_model_id",
                message=(
                    "Every node is priced on the frontier model. Routing the worker roles to "
                    "a smaller one is usually the largest single saving available in an "
                    "agent design."
                ),
            )
        )

    latency_constrained = any(
        word in " ".join(constraints).lower() for word in ("latency", "real-time", "realtime")
    )
    if latency_constrained and coordination == "sequential":
        warnings.append(
            ToolWarning(
                level="warning",
                message=(
                    "A latency constraint and a sequential topology are in tension: nothing "
                    "overlaps, so the wait is the sum of every step. Parallel where the work "
                    "is genuinely independent."
                ),
            )
        )

    return warnings


def _document(
    *,
    goal: str,
    coordination: str,
    summary: str,
    diagram: str,
    node_rows: list[dict[str, Any]],
    contracts: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    framework: ToolOut | None,
    per_task: Decimal,
    per_month: Decimal,
    constraints: list[str],
) -> str:
    node_table = "\n".join(
        f"| {row['node']} | {row['model']} | {row['steps']} | {row['tools']} | "
        f"{row['cost_per_task']} | {row['responsibility']} |"
        for row in node_rows
    )
    contract_table = "\n".join(
        f"| {row['from']} → {row['to']} | {row['contract']} |" for row in contracts
    )
    failure_table = "\n".join(
        f"| {row['mode']} | {row['likelihood']} | {row['mitigation']} |" for row in failures
    )
    constraint_lines = "\n".join(f"- {item}" for item in constraints) or "- None stated."

    return f"""# Multi-agent workflow plan

{summary}

## Goal

{goal.strip()}

## Constraints

{constraint_lines}

## Topology — {coordination}

```mermaid
{diagram}
```

## Agents

| Node | Model | Steps | Tools | Cost/task | Responsibility |
| --- | --- | --- | --- | --- | --- |
{node_table}

Total: **{_usd(_money(per_task))} per task**, **{_usd(_display(per_month))} per month** at the
stated volume. Token profiles are defaults — enter measured ones in the agent cost
calculator once the loop is running.

## Handoff contracts

Every edge is a contract. An agent system fails at its edges far more often than
inside its agents, and an unwritten contract is the reason.

| Edge | Contract |
| --- | --- |
{contract_table}

## Failure modes

| Mode | Likelihood | Mitigation |
| --- | --- | --- |
{failure_table}

## Framework

{
        f"**{framework.name}** — {framework.description}"
        if framework
        else "No catalog framework is currently recommendable for this shape; implement directly "
        "against a provider SDK."
    }

---

Generated by StackForge. Topology, contracts, costs, and failure modes are
rule-based against the stated coordination style and tool list.
"""
