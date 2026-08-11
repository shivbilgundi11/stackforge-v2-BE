"""Serialisation, the bundle, the PDF, and plan gating (M18).

The bar these tests hold is FR-11: re-exporting the same thing produces
byte-identical output. Every format is checked against it, including the zip
and the PDF, because "deterministic except for the two hard ones" is not a
guarantee anyone can rely on.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile

import pytest
import yaml
from pypdf import PdfReader

from app.api.deps import Identity
from app.core.errors import PlanRequired, ValidationFailed
from app.models.export import ExportFormat
from app.models.user import Plan, User
from app.services import export_service, pdf_service
from tests.unit.test_artifact_generators import run_source, stack_source


def identity(plan: Plan = Plan.FREE, *, anonymous: bool = False) -> Identity:
    if anonymous:
        return Identity(user=None, anonymous_id="anon_test", session_id=None)
    user = User(id="usr_test", email="ada@example.com", plan=plan)
    return Identity(user=user, anonymous_id=None, session_id=None)


def render(source, export_format: ExportFormat, **kwargs):  # type: ignore[no-untyped-def]
    return export_service.render(source, export_format=export_format, **kwargs)


# ── plan gating ──────────────────────────────────────────────────────────────


def test_markdown_is_free_and_every_other_format_is_pro() -> None:
    assert export_service.required_plan(ExportFormat.MARKDOWN) is Plan.FREE
    for export_format in ExportFormat:
        if export_format is not ExportFormat.MARKDOWN:
            assert export_service.required_plan(export_format) is Plan.PRO


def test_an_anonymous_caller_can_export_markdown() -> None:
    """Running and exporting are free; *keeping* is what the account is for."""
    export_service.assert_allowed(ExportFormat.MARKDOWN, identity(anonymous=True))


def test_a_free_user_asking_for_pdf_gets_the_upgrade_details() -> None:
    with pytest.raises(PlanRequired) as raised:
        export_service.assert_allowed(ExportFormat.PDF, identity(Plan.FREE))

    assert raised.value.http_status == 402
    assert raised.value.details == {
        "required_plan": "pro",
        "current_plan": "free",
        # M20: the dialog branches on this to offer signup rather than a
        # billing page. False here — this caller has an account, just not the
        # plan.
        "requires_account": False,
        "format": "pdf",
    }


def test_an_anonymous_caller_asking_for_pdf_is_told_to_sign_up_first() -> None:
    """Two different walls, two different buttons. Sending someone without an
    account to a billing page is sending them to a page they cannot use."""
    with pytest.raises(PlanRequired) as raised:
        export_service.assert_allowed(ExportFormat.PDF, identity(anonymous=True))

    assert raised.value.details is not None
    assert raised.value.details["requires_account"] is True


def test_a_team_plan_clears_a_pro_gate() -> None:
    export_service.assert_allowed(ExportFormat.ZIP, identity(Plan.TEAM))


# ── markdown ─────────────────────────────────────────────────────────────────


def test_markdown_of_a_stack_is_the_architecture_document() -> None:
    rendered = render(stack_source(), ExportFormat.MARKDOWN)

    assert rendered.filename == "client-x-rag-rollout.md"
    assert rendered.content_type.startswith("text/markdown")
    assert b"# Stack architecture" in rendered.data


def test_markdown_of_a_non_markdown_artifact_is_wrapped_not_renamed() -> None:
    """A YAML file under a `.md` name would render as one long paragraph."""
    rendered = render(stack_source(), ExportFormat.MARKDOWN, artifact_type="compose")
    text = rendered.data.decode()

    assert text.startswith("# Docker Compose")
    assert "```yaml" in text
    assert rendered.filename == "client-x-rag-rollout-compose.md"


# ── json / yaml ──────────────────────────────────────────────────────────────


def test_the_json_envelope_is_self_describing() -> None:
    payload = json.loads(render(stack_source(), ExportFormat.JSON).data)

    assert payload["stackforge"]["schema"] == "stackforge.export/v1"
    assert payload["stackforge"]["source_type"] == "stack"
    assert payload["stack"]["version"] == 3
    assert len(payload["stack"]["components"]) == 7


def test_the_envelope_carries_no_wall_clock_stamp() -> None:
    """A `generated_at: now()` would break byte-identity on the second export.

    `source_updated_at` answers the question a reader of an old file actually
    has — how old is the thing this describes — and is stable.
    """
    meta = json.loads(render(stack_source(), ExportFormat.JSON).data)["stackforge"]

    assert "generated_at" not in meta
    assert meta["source_updated_at"] == "2026-06-29T12:00:00+00:00"


def test_yaml_and_json_describe_the_same_thing() -> None:
    as_json = json.loads(render(stack_source(), ExportFormat.JSON).data)
    as_yaml = yaml.safe_load(render(stack_source(), ExportFormat.YAML).data)
    assert as_json == as_yaml


def test_a_run_export_carries_the_full_wire_shape() -> None:
    payload = json.loads(render(run_source(), ExportFormat.JSON).data)

    assert payload["result"]["run_id"] == "run_fixture"
    assert payload["result"]["metrics"]["monthly_hours"] == "192.00"
    assert payload["input"]["team_size"] == 12


# ── csv ──────────────────────────────────────────────────────────────────────


def test_csv_of_a_single_table_needs_no_table_name() -> None:
    rendered = render(run_source(), ExportFormat.CSV)
    rows = list(csv.DictReader(io.StringIO(rendered.data.decode())))

    assert rows == [{"activity": "Triage", "hours_per_month": "96.00"}]


def test_csv_with_several_tables_asks_which_one() -> None:
    """Picking one silently would produce a file whose contents depend on dict
    ordering."""
    with pytest.raises(ValidationFailed) as raised:
        render(stack_source(), ExportFormat.CSV)

    assert "components" in raised.value.message
    assert raised.value.details is not None
    assert raised.value.details["fields"][0]["path"] == "table"


def test_csv_uses_rfc4180_line_endings() -> None:
    """Left to the platform, the same export differs between machines."""
    data = render(stack_source(), ExportFormat.CSV, table="score_breakdown").data
    assert data.count(b"\r\n") == 11  # header plus ten dimensions


def test_csv_of_a_result_with_nothing_tabular_refuses() -> None:
    source = run_source()
    source.output.tables.clear()

    with pytest.raises(ValidationFailed):
        render(source, ExportFormat.CSV)


# ── the bundle ───────────────────────────────────────────────────────────────


EXPECTED_BUNDLE = {
    "client-x-rag-rollout-plan/README.md",
    "client-x-rag-rollout-plan/architecture.md",
    "client-x-rag-rollout-plan/architecture.mmd",
    "client-x-rag-rollout-plan/cost-estimate.md",
    "client-x-rag-rollout-plan/roadmap.md",
    "client-x-rag-rollout-plan/deploy/docker-compose.yml",
    "client-x-rag-rollout-plan/deploy/.env.example",
    "client-x-rag-rollout-plan/.cursorrules",
}


def test_the_bundle_contains_exactly_the_expected_files() -> None:
    with zipfile.ZipFile(io.BytesIO(render(stack_source(), ExportFormat.ZIP).data)) as archive:
        assert set(archive.namelist()) == EXPECTED_BUNDLE


def test_the_bundle_unzips_cleanly_and_the_readme_lists_what_is_in_it() -> None:
    with zipfile.ZipFile(io.BytesIO(render(stack_source(), ExportFormat.ZIP).data)) as archive:
        assert archive.testzip() is None
        readme = archive.read("client-x-rag-rollout-plan/README.md").decode()

    for path in EXPECTED_BUNDLE:
        assert f"`{path.removeprefix('client-x-rag-rollout-plan/')}`" in readme
    assert "starter template" in readme


def test_a_deprecated_component_is_flagged_at_the_top_of_the_bundle() -> None:
    from tests.unit.test_artifact_generators import tool

    source = stack_source(
        [
            tool("chroma", category="vector-db", status="caution"),
            tool("llamaindex", category="rag-framework"),
        ]
    )
    with zipfile.ZipFile(io.BytesIO(render(source, ExportFormat.ZIP).data)) as archive:
        readme = archive.read("client-x-rag-rollout-plan/README.md").decode()

    assert "marked\ndeprecated or caution" in readme or "deprecated or caution" in readme
    assert "Chroma" in readme


# ── pdf ──────────────────────────────────────────────────────────────────────


def test_the_pdf_is_valid_and_contains_the_documents_text() -> None:
    data = render(stack_source(), ExportFormat.PDF).data
    assert data.startswith(b"%PDF-")

    reader = PdfReader(io.BytesIO(data))
    assert len(reader.pages) >= 2  # a cover page, then the body
    text = "\n".join(page.extract_text() for page in reader.pages)

    assert "Client X RAG rollout" in text
    assert "Qdrant" in text


def test_the_pdf_footer_carries_the_share_url() -> None:
    """`PRD.md` §24 makes the share URL a retention mechanic: the recipient's
    copy has to lead back here."""
    data = export_service.render(
        stack_source(),
        export_format=ExportFormat.PDF,
        share_url="https://stackforge.dev/s/tok",
    ).data
    reader = PdfReader(io.BytesIO(data))

    assert "stackforge.dev/s/tok" in "\n".join(page.extract_text() for page in reader.pages)


def test_the_reportlab_backend_is_byte_identical_across_renders() -> None:
    document = pdf_service.Document(
        title="A plan",
        subtitle="Architecture",
        markdown="# Heading\n\nBody text.\n",
        generated_at=stack_source().updated_at,
    )
    first = pdf_service.render(document, backend="reportlab")
    second = pdf_service.render(document, backend="reportlab")

    assert first == second


def test_chromium_is_the_default_when_playwright_is_present(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(pdf_service, "_playwright_available", lambda: True)
    assert pdf_service._resolve_backend("auto") == "chromium"


def test_an_explicit_chromium_request_falls_back_rather_than_failing(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A worse PDF beats no PDF for a user who has already paid for it."""
    monkeypatch.setattr(pdf_service, "_playwright_available", lambda: False)
    assert pdf_service._resolve_backend("chromium") == "reportlab"


# ── idempotency (FR-11) ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "export_format",
    [ExportFormat.MARKDOWN, ExportFormat.JSON, ExportFormat.YAML, ExportFormat.ZIP],
)
def test_re_exporting_produces_identical_bytes(export_format: ExportFormat) -> None:
    assert render(stack_source(), export_format).data == render(stack_source(), export_format).data


def test_re_exporting_csv_produces_identical_bytes() -> None:
    first = render(stack_source(), ExportFormat.CSV, table="components")
    second = render(stack_source(), ExportFormat.CSV, table="components")
    assert first.data == second.data


def test_re_exporting_pdf_produces_identical_bytes() -> None:
    """Asserted against the ReportLab backend, which is put into invariant
    mode. Chromium stamps its own creation date into the trailer and offers no
    way to suppress it, so claiming this of Chromium would not be true."""
    from app.core.config import settings

    original = settings.pdf_backend
    settings.pdf_backend = "reportlab"
    try:
        assert render(stack_source(), ExportFormat.PDF).data == (
            render(stack_source(), ExportFormat.PDF).data
        )
    finally:
        settings.pdf_backend = original


# ── the async threshold ──────────────────────────────────────────────────────


def test_only_bundles_are_ever_queued() -> None:
    """A 2 KB Markdown export going through a queue would be a worse
    experience for no reason."""
    source = stack_source()
    for export_format in ExportFormat:
        if export_format is not ExportFormat.ZIP:
            assert export_service.predicted_bytes(source, export_format) == 0
            assert not export_service.should_queue(source, export_format)


def test_a_bundle_is_queued_once_it_is_predicted_to_be_large() -> None:
    from app.core.config import settings

    source = stack_source()
    assert not export_service.should_queue(source, ExportFormat.ZIP)

    original = settings.export_async_threshold_bytes
    settings.export_async_threshold_bytes = 1_000
    try:
        assert export_service.should_queue(source, ExportFormat.ZIP)
    finally:
        settings.export_async_threshold_bytes = original
