"""Markdown to PDF, behind a swappable backend.

**Q-01, answered here.** Two backends, one interface:

  * `chromium` — renders the Markdown through a styled HTML template with
    Playwright. This is the recommended production path and the default when
    it is available. "A PDF I can send to my client" is the thing being sold,
    and a PDF that looks like a generated report undercuts exactly that; real
    typography, real page breaks, and a real cover page need a browser. It
    costs a ~400 MB image, which is why it runs as its own worker service and
    is not a hard dependency of the API.
  * `reportlab` — no browser, no image weight, honest output. Headings,
    tables, code blocks, lists, and rules all render; what it does not do is
    look designed.

`auto` picks Chromium when Playwright is importable and falls back to
ReportLab otherwise, logging which one it chose. The fallback is loud rather
than silent: a deploy that quietly started producing worse PDFs than the one
before it is the failure mode this whole module is arranged to avoid, and a
log line is the cheapest thing that catches it.

Both backends start from the same Markdown, so switching one for the other
changes how a document looks and never what it says.

**One limit worth stating.** FR-11's byte-identity guarantee holds for the
ReportLab backend, which is put into invariant mode so nothing wall-clock
reaches the file. Chromium stamps its own creation date into the PDF trailer
and offers no way to suppress it, so two Chromium renders of the same document
differ in bytes while being identical in content. The idempotency test
therefore pins ReportLab explicitly rather than asserting something about
Chromium that is not true.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import Any, Final, Literal

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.core.config import settings
from app.core.database import utcnow
from app.core.logging import get_logger
from app.services import diagram_render

logger = get_logger("pdf")

Backend = Literal["auto", "chromium", "reportlab"]

#: How long the diagram pass gets. Generous — a stack diagram is eight boxes
#: and renders in well under a second — but bounded, because the alternative to
#: a bound is an export that never returns.
DIAGRAM_TIMEOUT_MS: Final = 15_000

#: Forge Console, as close as print gets. The palette is the light theme —
#: a PDF is printed or read on white, and a dark document is a document that
#: empties a toner cartridge.
INK: Final = "#1c1917"
MUTED: Final = "#78716c"
ACCENT: Final = "#c2410c"
RULE: Final = "#e7e5e4"
SURFACE: Final = "#fafaf9"


@dataclass(frozen=True)
class Document:
    """What a PDF needs beyond its body.

    The cover exists because the artifact is handed to someone who was not in
    the room. A document that opens on "## Components" with no statement of
    what it is or when it was made is a document that gets asked about rather
    than read.
    """

    title: str
    subtitle: str
    markdown: str
    #: Rendered into the footer of every page. `PRD.md` §24 makes the share URL
    #: a retention mechanic: the recipient's copy has to lead back here.
    share_url: str | None = None
    generated_at: datetime | None = None

    @property
    def stamped_at(self) -> datetime:
        return self.generated_at or utcnow()


def render(document: Document, *, backend: Backend | None = None) -> bytes:
    chosen = _resolve_backend(backend or settings.pdf_backend)
    if chosen == "chromium":
        return _render_chromium(document)
    return _render_reportlab(document)


def _resolve_backend(requested: Backend) -> Literal["chromium", "reportlab"]:
    if requested == "reportlab":
        return "reportlab"
    if _playwright_available():
        return "chromium"
    if requested == "chromium":
        # Asked for explicitly and not installed. Falling back anyway rather
        # than failing the export: a worse PDF is a better outcome than no PDF
        # for a user who has already paid for the feature.
        logger.warning("pdf.chromium_unavailable", fallback="reportlab")
    else:
        logger.info("pdf.backend_selected", backend="reportlab", reason="playwright not installed")
    return "reportlab"


def _playwright_available() -> bool:
    try:
        import playwright.sync_api  # noqa: F401
    except ImportError:
        return False
    return True


# ── shared: markdown to HTML ────────────────────────────────────────────────


#: What `fenced_code` makes of a ```mermaid block. Rewritten to the shape the
#: renderer looks for, and unescaped on the way — a diagram's labels carry
#: `<br/>` and quotes, which the Markdown pass turns into entities that Mermaid
#: would then try to parse as part of a node name.
_MERMAID_BLOCK = re.compile(r'<pre><code class="language-mermaid">(.*?)</code></pre>', re.DOTALL)


def _mermaid_blocks(body: str) -> str:
    return _MERMAID_BLOCK.sub(
        lambda match: f'<pre class="mermaid">{html.unescape(match.group(1))}</pre>',
        body,
    )


def to_html(document: Document) -> str:
    """The styled HTML both the Chromium backend and a browser preview use."""
    import markdown as markdown_lib

    body = markdown_lib.markdown(
        document.markdown,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
        output_format="html",
    )
    # Only Chromium can turn these into a picture; ReportLab never sees this
    # function. Marking them here rather than in the backend keeps the one
    # place that knows Markdown in charge of reading it.
    body = _mermaid_blocks(body)
    stamped = document.stamped_at.strftime("%d %B %Y")
    footer = html.escape(document.share_url or "stackforge.dev")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(document.title)}</title>
<style>
  @page {{
    size: A4;
    margin: 22mm 18mm 20mm;
    @bottom-left {{ content: "{footer}"; }}
    @bottom-right {{ content: counter(page); }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, sans-serif;
    color: {INK};
    font-size: 10.5pt;
    line-height: 1.55;
    margin: 0;
  }}
  .cover {{
    page-break-after: always;
    padding-top: 45mm;
    border-top: 4px solid {ACCENT};
  }}
  .cover h1 {{ font-size: 30pt; line-height: 1.15; margin: 0 0 8pt; letter-spacing: -0.5pt; }}
  .cover p {{ color: {MUTED}; font-size: 12pt; margin: 0 0 4pt; }}
  .cover .stamp {{ margin-top: 40mm; font-size: 9pt; color: {MUTED}; }}
  h1, h2, h3, h4 {{ color: {INK}; line-height: 1.25; margin: 18pt 0 6pt; }}
  h1 {{ font-size: 19pt; border-bottom: 1px solid {RULE}; padding-bottom: 5pt; }}
  h2 {{ font-size: 14pt; }}
  h3 {{ font-size: 11.5pt; }}
  p, li {{ margin: 0 0 7pt; }}
  a {{ color: {ACCENT}; text-decoration: none; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 8pt 0 12pt;
    font-size: 9pt;
    page-break-inside: avoid;
  }}
  th {{
    text-align: left;
    background: {SURFACE};
    border-bottom: 1.5px solid {RULE};
    padding: 5pt 6pt;
    font-weight: 600;
  }}
  td {{ border-bottom: 1px solid {RULE}; padding: 5pt 6pt; vertical-align: top; }}
  pre {{
    background: {SURFACE};
    border: 1px solid {RULE};
    border-radius: 3pt;
    padding: 8pt;
    font-size: 8.5pt;
    white-space: pre-wrap;
    word-break: break-word;
    page-break-inside: avoid;
  }}
  code {{ font-family: "Cascadia Mono", Consolas, monospace; font-size: 9pt; }}
  /* Until the renderer replaces it, a diagram is still its source — so it is
     styled as one. If Mermaid is unavailable the block simply stays. */
  pre.mermaid {{ white-space: pre; }}
  figure.diagram {{
    margin: 10pt 0 14pt;
    padding: 8pt;
    border: 1px solid {RULE};
    border-radius: 3pt;
    text-align: center;
    page-break-inside: avoid;
  }}
  figure.diagram svg {{ max-width: 100%; height: auto; }}
  hr {{ border: 0; border-top: 1px solid {RULE}; margin: 14pt 0; }}
  blockquote {{
    margin: 8pt 0;
    padding-left: 10pt;
    border-left: 3px solid {ACCENT};
    color: {MUTED};
  }}
</style>
</head>
<body>
<section class="cover">
  <h1>{html.escape(document.title)}</h1>
  <p>{html.escape(document.subtitle)}</p>
  <div class="stamp">
    Generated by StackForge on {stamped}<br>
    {footer}
  </div>
</section>
{body}
</body>
</html>
"""


# ── chromium ────────────────────────────────────────────────────────────────


def _render_chromium(document: Document) -> bytes:
    from playwright.sync_api import sync_playwright

    markup = to_html(document)
    footer = html.escape(document.share_url or "stackforge.dev")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=["--no-sandbox"])
        try:
            page = browser.new_page()
            # `set_content` with a data-free document rather than a temp file:
            # nothing here loads a remote resource, so there is no network to
            # wait for and no file to clean up on a crash.
            page.set_content(markup, wait_until="load")
            _draw_diagrams(page)
            rendered: bytes = page.pdf(
                format="A4",
                print_background=True,
                display_header_footer=True,
                header_template="<div></div>",
                footer_template=(
                    '<div style="width:100%;font-size:8px;color:#78716c;'
                    'padding:0 18mm;display:flex;justify-content:space-between;">'
                    f"<span>{footer}</span>"
                    '<span class="pageNumber"></span></div>'
                ),
                margin={"top": "20mm", "bottom": "18mm", "left": "18mm", "right": "18mm"},
            )
            return rendered
        finally:
            browser.close()


def _draw_diagrams(page: Any) -> None:
    """Turn the diagram sources in the page into pictures, or leave them.

    Everything here is best-effort by construction. The bundle is added from
    disk rather than a URL, so there is still no network in an export; if it is
    missing, or Mermaid throws, or the render outlives its budget, the page
    keeps the fenced source it already had and the PDF is the one this backend
    produced last week.
    """
    if not diagram_render.available():
        return

    try:
        page.add_script_tag(path=str(diagram_render.MERMAID_JS))
        page.add_script_tag(content=diagram_render.script())
        # A budget rather than an open wait. A diagram that cannot finish must
        # cost the picture, not the export.
        drawn = page.evaluate("window.__sfDiagramsReady", timeout=DIAGRAM_TIMEOUT_MS)
        logger.info("pdf.diagrams_drawn", count=drawn)
    except Exception as exc:  # any failure degrades to the fenced source
        logger.warning("pdf.diagrams_failed", error=str(exc))


# ── reportlab ───────────────────────────────────────────────────────────────

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_LIST_ITEM = re.compile(r"^\s*[-*+]\s+(.*)$")
_ORDERED_ITEM = re.compile(r"^\s*(\d+)[.)]\s+(.*)$")
_TABLE_DIVIDER = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<!\*)\*([^*]+?)\*(?!\*)")
_CODE = re.compile(r"`([^`]+?)`")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def _inline(text: str) -> str:
    """Markdown inline spans to ReportLab's mini-markup.

    Escaped *first*. A component description containing `<` would otherwise
    open a tag ReportLab cannot close, and the whole paragraph would fail to
    parse — taking the export with it.
    """
    escaped = html.escape(text, quote=False)
    escaped = _CODE.sub(r'<font face="Courier">\1</font>', escaped)
    escaped = _BOLD.sub(r"<b>\1</b>", escaped)
    escaped = _ITALIC.sub(r"<i>\1</i>", escaped)
    escaped = _LINK.sub(rf'<font color="{ACCENT}">\1</font>', escaped)
    return escaped


def _render_reportlab(document: Document) -> bytes:
    buffer = BytesIO()
    base = getSampleStyleSheet()
    ink = colors.HexColor(INK)
    muted = colors.HexColor(MUTED)
    rule = colors.HexColor(RULE)

    styles = {
        "title": ParagraphStyle(
            "sf-title",
            parent=base["Title"],
            fontSize=26,
            leading=30,
            textColor=ink,
            alignment=TA_LEFT,
        ),
        "subtitle": ParagraphStyle(
            "sf-subtitle", parent=base["Normal"], fontSize=12, leading=17, textColor=muted
        ),
        "stamp": ParagraphStyle(
            "sf-stamp", parent=base["Normal"], fontSize=8.5, leading=13, textColor=muted
        ),
        "h1": ParagraphStyle(
            "sf-h1", parent=base["Heading1"], fontSize=17, leading=21, spaceBefore=14, textColor=ink
        ),
        "h2": ParagraphStyle(
            "sf-h2", parent=base["Heading2"], fontSize=13, leading=17, spaceBefore=12, textColor=ink
        ),
        "h3": ParagraphStyle(
            "sf-h3", parent=base["Heading3"], fontSize=11, leading=15, spaceBefore=10, textColor=ink
        ),
        "body": ParagraphStyle(
            "sf-body", parent=base["BodyText"], fontSize=9.5, leading=14, textColor=ink
        ),
        "bullet": ParagraphStyle(
            "sf-bullet",
            parent=base["BodyText"],
            fontSize=9.5,
            leading=14,
            leftIndent=10,
            bulletIndent=2,
            textColor=ink,
        ),
        "code": ParagraphStyle(
            "sf-code",
            parent=base["Code"],
            fontSize=8,
            leading=11,
            backColor=colors.HexColor(SURFACE),
            borderPadding=6,
            textColor=ink,
        ),
        "cell": ParagraphStyle(
            "sf-cell", parent=base["BodyText"], fontSize=8, leading=11, textColor=ink
        ),
        "cell-head": ParagraphStyle(
            "sf-cell-head",
            parent=base["BodyText"],
            fontSize=8,
            leading=11,
            textColor=ink,
            fontName="Helvetica-Bold",
        ),
    }

    story: list[Any] = [
        Spacer(1, 45 * mm),
        Paragraph(_inline(document.title), styles["title"]),
        Spacer(1, 4 * mm),
        Paragraph(_inline(document.subtitle), styles["subtitle"]),
        Spacer(1, 40 * mm),
        Paragraph(
            f"Generated by StackForge on {document.stamped_at.strftime('%d %B %Y')}<br/>"
            f"{html.escape(document.share_url or 'stackforge.dev')}",
            styles["stamp"],
        ),
        PageBreak(),
    ]

    for block in _blocks(document.markdown):
        kind = block["kind"]
        if kind == "heading":
            level = min(3, int(block["level"]))
            story.append(Paragraph(_inline(str(block["text"])), styles[f"h{level}"]))
            if level == 1:
                story.append(HRFlowable(width="100%", color=rule, spaceAfter=4))
        elif kind == "paragraph":
            story.append(Paragraph(_inline(str(block["text"])), styles["body"]))
            story.append(Spacer(1, 3))
        elif kind == "bullet":
            story.append(Paragraph(_inline(str(block["text"])), styles["bullet"], bulletText="•"))
        elif kind == "ordered":
            story.append(
                Paragraph(
                    _inline(str(block["text"])), styles["bullet"], bulletText=f"{block['index']}."
                )
            )
        elif kind == "code":
            story.append(
                Paragraph(
                    html.escape(str(block["text"])).replace("\n", "<br/>").replace(" ", "&nbsp;"),
                    styles["code"],
                )
            )
            story.append(Spacer(1, 6))
        elif kind == "rule":
            story.append(HRFlowable(width="100%", color=rule, spaceBefore=8, spaceAfter=8))
        elif kind == "table":
            story.append(KeepTogether(_table(block, styles, rule)))
            story.append(Spacer(1, 6))

    template = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=20 * mm,
        bottomMargin=18 * mm,
        title=document.title,
        author="StackForge",
        # Fixes the document id and the embedded creation date. Without it
        # ReportLab stamps `now()` into the trailer and two renders of the same
        # input differ in bytes — which is FR-11's idempotency requirement
        # failing for a reason that has nothing to do with the content.
        invariant=1,
    )
    footer_text = document.share_url or "stackforge.dev"

    def _decorate(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(muted)
        canvas.drawString(18 * mm, 11 * mm, footer_text)
        canvas.drawRightString(A4[0] - 18 * mm, 11 * mm, str(doc.page))
        canvas.restoreState()

    template.build(story, onFirstPage=_decorate, onLaterPages=_decorate)
    return buffer.getvalue()


def _table(block: dict[str, Any], styles: dict[str, Any], rule: Any) -> Any:
    header = [Paragraph(_inline(cell), styles["cell-head"]) for cell in block["header"]]
    rows = [[Paragraph(_inline(cell), styles["cell"]) for cell in row] for row in block["rows"]]

    # `repeatRows=1` so a table that spans a page break carries its header
    # onto the next page. Without it a long components table becomes a wall of
    # unlabelled cells from page two onward.
    table = Table([header, *rows], repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(SURFACE)),
                ("LINEBELOW", (0, 0), (-1, 0), 1, rule),
                ("LINEBELOW", (0, 1), (-1, -1), 0.5, rule),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _blocks(markdown_text: str) -> list[dict[str, Any]]:
    """Markdown into flat, typed blocks.

    A deliberately small subset — the exact subset the generators in
    `services/artifacts/` emit. Handling more would be handling input this
    codebase never produces, and every unhandled construct here is one a
    generator is not allowed to use.
    """
    blocks: list[dict[str, Any]] = []
    lines = markdown_text.replace("\r\n", "\n").split("\n")
    index = 0

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            index += 1
            continue

        if stripped.startswith("```"):
            fence: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                fence.append(lines[index])
                index += 1
            index += 1
            blocks.append({"kind": "code", "text": "\n".join(fence)})
            continue

        if stripped in {"---", "***", "___"}:
            blocks.append({"kind": "rule"})
            index += 1
            continue

        heading = _HEADING.match(stripped)
        if heading:
            blocks.append(
                {"kind": "heading", "level": len(heading.group(1)), "text": heading.group(2)}
            )
            index += 1
            continue

        # A table is a pipe row followed by a divider row. Checking the divider
        # rather than the pipe alone matters: prose containing a pipe is not a
        # table, and treating it as one swallows the paragraph after it.
        if (
            stripped.startswith("|")
            and index + 1 < len(lines)
            and _TABLE_DIVIDER.match(lines[index + 1].strip())
        ):
            header = _cells(stripped)
            index += 2
            rows: list[list[str]] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                rows.append(_cells(lines[index].strip()))
                index += 1
            width = len(header)
            blocks.append(
                {
                    "kind": "table",
                    "header": header,
                    # Ragged rows are padded rather than dropped. A generator
                    # emitting one is a bug, but losing the row hides it.
                    "rows": [(row + [""] * width)[:width] for row in rows],
                }
            )
            continue

        bullet = _LIST_ITEM.match(line)
        if bullet:
            blocks.append({"kind": "bullet", "text": bullet.group(1)})
            index += 1
            continue

        ordered = _ORDERED_ITEM.match(line)
        if ordered:
            blocks.append({"kind": "ordered", "index": ordered.group(1), "text": ordered.group(2)})
            index += 1
            continue

        paragraph = [stripped]
        index += 1
        while index < len(lines) and lines[index].strip() and not _breaks(lines[index]):
            paragraph.append(lines[index].strip())
            index += 1
        blocks.append({"kind": "paragraph", "text": " ".join(paragraph)})

    return blocks


def _breaks(line: str) -> bool:
    stripped = line.strip()
    return bool(
        stripped.startswith(("#", "|", "```", "- ", "* ", "+ "))
        or stripped in {"---", "***", "___"}
        or _ORDERED_ITEM.match(line)
    )


def _cells(row: str) -> list[str]:
    parts = row.strip().strip("|").split("|")
    return [part.strip() for part in parts]


__all__ = ["Backend", "Document", "render", "to_html"]
