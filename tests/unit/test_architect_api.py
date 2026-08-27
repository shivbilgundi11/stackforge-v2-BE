from __future__ import annotations

from decimal import Decimal
from typing import Any

from app.api.v1.architect import _apply_choice
from app.schemas.tools import ToolOutput
from app.services import stack_architect_service, stack_score_service
from app.services.stack_score_service import DIMENSIONS


def _candidate(rank: int) -> stack_architect_service.Candidate:
    return stack_architect_service.Candidate(
        rank=rank,
        components=[],
        score=stack_score_service.StackScore(
            total=Decimal(90 - rank),
            dimensions={dimension.key: Decimal(8) for dimension in DIMENSIONS},
        ),
        compatibility=None,
        deprecated=[],
    )


def _build(rebuilt: dict[int, ToolOutput]) -> Any:
    def build(winner: stack_architect_service.Candidate) -> ToolOutput:
        return rebuilt[winner.rank]

    return build


def test_the_models_pick_replaces_the_whole_result_not_part_of_it() -> None:
    """A swap that moved the components and left the diagram would put one
    stack's picture over another's numbers — which is the failure the single
    builder exists to prevent."""
    ranked = [_candidate(1), _candidate(2)]
    output = ToolOutput(metrics={"score": Decimal("89")}, tables={"components": [{"n": "leader"}]})
    runner_up = ToolOutput(
        metrics={"score": Decimal("88")}, tables={"components": [{"n": "runner-up"}]}
    )

    _apply_choice(output, "2", _build({2: runner_up}), ranked)

    assert output.metrics == runner_up.metrics
    assert output.tables == runner_up.tables


def test_an_unoffered_rank_leaves_the_engines_leader_in_place() -> None:
    """Anything the engine did not rank is a malformed answer, and the
    fallback for a malformed answer is the deterministic result (D-06)."""
    ranked = [_candidate(1), _candidate(2)]

    def build(winner: stack_architect_service.Candidate) -> ToolOutput:
        raise AssertionError("rebuilt on an answer that named nothing")

    for raw in (None, "", "nonsense", "3000", "1"):
        output = ToolOutput(metrics={"score": Decimal("89")})
        _apply_choice(output, raw, build, ranked)
        assert output.metrics == {"score": Decimal("89")}
