"""The diagram pass in the PDF backend.

What the browser does with the page cannot be asserted without a browser — the
Chromium backend is an optional install and CI does not have it. What *can* be
asserted is everything that decides what the browser is handed, which is where
this has already been wrong once: a Markdown pass that escapes a diagram's
labels hands Mermaid `&lt;br/&gt;` and it parses that as part of a node name.
"""

from __future__ import annotations

import json

from app.services import diagram_render, pdf_service

DIAGRAM = 'graph LR\n    llm["LLM provider<br/>Anthropic API"]\n    client --> llm'


def _document(markdown: str) -> pdf_service.Document:
    return pdf_service.Document(title="T", subtitle="S", markdown=markdown)


def test_a_mermaid_block_becomes_the_shape_the_renderer_looks_for() -> None:
    html = pdf_service.to_html(_document(f"# H\n\n```mermaid\n{DIAGRAM}\n```\n"))

    assert '<pre class="mermaid">' in html
    assert 'class="language-mermaid"' not in html


def test_the_diagram_source_is_unescaped_on_the_way() -> None:
    """The regression this exists for. `<br/>` in a label survives Markdown as
    an entity, and Mermaid reads the entity as text inside the node name."""
    html = pdf_service.to_html(_document(f"```mermaid\n{DIAGRAM}\n```\n"))

    assert "<br/>" in html
    assert "&lt;br/&gt;" not in html
    assert '"LLM provider<br/>Anthropic API"' in html


def test_an_ordinary_code_block_is_left_alone() -> None:
    html = pdf_service.to_html(_document("```python\nprint('hi')\n```\n"))

    assert 'class="language-python"' in html
    assert 'class="mermaid"' not in html


def test_a_document_with_no_diagram_gets_no_diagram_block() -> None:
    # The stylesheet always carries the rules; what must not appear is a block
    # for the renderer to pick up.
    html = pdf_service.to_html(_document("# Just prose\n\nA paragraph.\n"))

    assert '<pre class="mermaid">' not in html


def test_the_page_carries_a_style_for_the_rendered_figure() -> None:
    # Until the renderer replaces it the block is still source, and if Mermaid
    # never arrives it stays that way — both states are styled.
    html = pdf_service.to_html(_document(f"```mermaid\n{DIAGRAM}\n```\n"))

    assert "figure.diagram" in html
    assert "pre.mermaid" in html


def test_the_vendored_bundle_is_present() -> None:
    """Committed rather than fetched: an export must not depend on a CDN."""
    assert diagram_render.available()
    assert diagram_render.MERMAID_JS.stat().st_size > 1_000_000


def test_the_page_script_carries_the_marks_rather_than_fetching_them() -> None:
    script = diagram_render.script()

    assert "postgresql" in script
    assert "__sfDiagramsReady" in script
    # No network in an export, in the script as well as around it.
    assert "fetch(" not in script


def test_the_script_and_the_web_renderer_read_the_same_marks_file() -> None:
    """Two implementations, one data file. If these ever diverge the same
    diagram renders two ways in the app and in the PDF."""
    backend = json.loads(diagram_render.BRAND_MARKS.read_text(encoding="utf-8"))
    web = json.loads(
        (
            diagram_render.BRAND_MARKS.parents[3] / "frontend" / "lib" / "brand" / "marks.json"
        ).read_text(encoding="utf-8")
    )

    assert backend == web
