from __future__ import annotations

from decimal import Decimal

from app.api.v1.architect import _apply_model_scores
from app.schemas.tools import ToolOutput
from app.services.stack_score_service import DIMENSIONS


def test_gemini_scores_replace_the_headline_and_breakdown_together() -> None:
    output = ToolOutput(
        metrics={"score": Decimal("85.0")},
        tables={
            "score_breakdown": [
                {
                    "key": dimension.key,
                    "score": "8.0",
                    "weight_pct": str(dimension.weight * 100),
                    "contribution": "0.0",
                }
                for dimension in DIMENSIONS
            ]
        },
    )
    assessment = [
        {"key": dimension.key, "score": 6 + (index % 4)}
        for index, dimension in enumerate(DIMENSIONS)
    ]

    _apply_model_scores(output, assessment)

    total = sum(Decimal(row["contribution"]) for row in output.tables["score_breakdown"])
    assert output.metrics["score"] == total
    assert output.tables["score_breakdown"][0]["score"] == "6.0"
