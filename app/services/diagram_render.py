"""Rendering a Mermaid diagram inside the PDF.

The export used to ship the diagram as a fenced code block, which is what M18
decided and what was right while the only backend was ReportLab: a fenced block
is identical on every backend and still copy-pasteable. The Chromium backend is
a real browser, though, and the architecture document's whole argument is a
picture — so on that backend the block becomes the diagram.

## What is vendored, and why

`app/static/mermaid.min.js`, about three and a half megabytes. It is committed
rather than fetched because `_render_chromium` loads no remote resource on
purpose — an export must not depend on a CDN being reachable — and because
three megabytes next to the four hundred that Chromium already costs is not the
line worth drawing. `MERMAID_VERSION` records which build it is; the web app
resolves its own copy from `package.json`, and the two should be bumped
together or the same diagram renders two ways.

## What this does not do

The ReportLab fallback still prints the source. It has no browser, so there is
nothing to render with, and a PDF that silently drops the diagram would be
worse than one that shows the text it was drawn from.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Final

from app.core.logging import get_logger

logger = get_logger("diagrams")

_STATIC: Final = Path(__file__).resolve().parent.parent / "static"
MERMAID_JS: Final = _STATIC / "mermaid.min.js"
BRAND_MARKS: Final = Path(__file__).resolve().parent.parent / "data" / "brand_marks.json"

#: The page is white and the ink is dark, whatever the reader's app theme is.
#: A PDF has one background and it is paper.
PAPER: Final = "#ffffff"
INK: Final = "#18181b"
LINE: Final = "#a1a1aa"
SURFACE: Final = "#f4f4f5"


@lru_cache(maxsize=1)
def available() -> bool:
    """Whether the vendored bundle is present.

    Cached: this is asked once per export and the answer cannot change without
    a deploy. A missing file is not an error — the document falls back to the
    fenced source, which is what every backend did before this existed.
    """
    if MERMAID_JS.is_file():
        return True
    logger.warning("diagrams.mermaid_missing", path=str(MERMAID_JS))
    return False


@lru_cache(maxsize=1)
def _marks_json() -> str:
    if not BRAND_MARKS.is_file():
        logger.warning("diagrams.brand_marks_missing", path=str(BRAND_MARKS))
        return "{}"
    return BRAND_MARKS.read_text(encoding="utf-8").strip()


def script() -> str:
    """The page script that turns every `pre.mermaid` into a diagram.

    A twin of `lib/brand/badges.ts` in the web app, and deliberately so: the
    two renderers draw the same badge from the same `%% brand:` metadata and
    the same generated `brand_marks.json`, but one runs inside React and the
    other inside a page this module builds by hand. The *data* is shared; these
    hundred lines are not, and a change to the badge shape belongs in both.

    `window.__sfDiagramsReady` is what the caller waits on. Resolving it even
    when a diagram fails is intentional — a broken diagram must not hang an
    export that is otherwise finished.
    """
    return f"""
const MARKS = {_marks_json()};
const PAPER = {json.dumps(PAPER)};
const BRAND_LINE = /^\\s*%%\\s*brand:([A-Za-z0-9_]+):([a-z0-9]*):([0-9a-fA-F]{{6}})\\s*$/;
const SVG_NS = "http://www.w3.org/2000/svg";
const RADIUS = 11;
const GLYPH = 13;

function readBrands(source) {{
  const out = new Map();
  for (const line of source.split("\\n")) {{
    const match = BRAND_LINE.exec(line);
    if (match) out.set(match[1], {{ icon: match[2], hex: match[3] }});
  }}
  return out;
}}

function luminance(hex) {{
  const value = parseInt(hex.replace("#", ""), 16);
  const parts = [(value >> 16) & 255, (value >> 8) & 255, value & 255].map((channel) => {{
    const scaled = channel / 255;
    return scaled <= 0.03928 ? scaled / 12.92 : Math.pow((scaled + 0.055) / 1.055, 2.4);
  }});
  return 0.2126 * parts[0] + 0.7152 * parts[1] + 0.0722 * parts[2];
}}

function contrast(a, b) {{
  const sorted = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (sorted[0] + 0.05) / (sorted[1] + 0.05);
}}

function badgeColours(brandHex, surface) {{
  const brand = brandHex.charAt(0) === "#" ? brandHex : "#" + brandHex;
  if (contrast(brand, surface) < 1.6) {{
    return {{ disc: luminance(surface) > 0.5 ? "#18181b" : "#e4e4e7", ink: brand }};
  }}
  return {{ disc: brand, ink: contrast(brand, "#ffffff") >= 2.4 ? "#ffffff" : "#18181b" }};
}}

function nodeKey(id) {{
  const index = id.indexOf("-flowchart-");
  if (index === -1) return null;
  return id.slice(index + "-flowchart-".length).replace(/-\\d+$/, "");
}}

function paintBadges(svg, source) {{
  const brands = readBrands(source);
  if (brands.size === 0) return;

  for (const node of Array.prototype.slice.call(svg.querySelectorAll("g.node"))) {{
    const key = nodeKey(node.id || "");
    const brand = key ? brands.get(key) : null;
    if (!key || !brand) continue;

    const box = node.querySelector("rect.label-container, rect.basic");
    if (!box) continue;
    const x = Number(box.getAttribute("x"));
    const y = Number(box.getAttribute("y"));
    if (!isFinite(x) || !isFinite(y)) continue;

    const mark = MARKS[brand.icon];
    const colours = badgeColours(mark ? mark.hex : brand.hex, PAPER);

    const badge = document.createElementNS(SVG_NS, "g");
    badge.setAttribute("transform", "translate(" + x + ", " + y + ")");

    const disc = document.createElementNS(SVG_NS, "circle");
    disc.setAttribute("r", String(RADIUS));
    // Painted through `style`, never the `fill` attribute: mermaid ships a
    // stylesheet inside the SVG that paints `.node circle` and `.node path`,
    // and a presentation attribute loses to any CSS rule.
    disc.style.fill = colours.disc;
    disc.style.stroke = PAPER;
    disc.style.strokeWidth = "2px";
    badge.appendChild(disc);

    if (mark) {{
      const glyph = document.createElementNS(SVG_NS, "path");
      glyph.setAttribute("d", mark.path);
      glyph.setAttribute(
        "transform",
        "translate(" + -GLYPH / 2 + ", " + -GLYPH / 2 + ") scale(" + GLYPH / 24 + ")",
      );
      glyph.style.fill = colours.ink;
      glyph.style.stroke = "none";
      badge.appendChild(glyph);
    }} else {{
      const letter = document.createElementNS(SVG_NS, "text");
      letter.setAttribute("x", "0");
      letter.setAttribute("y", "0");
      letter.setAttribute("dy", "0.36em");
      letter.setAttribute("text-anchor", "middle");
      letter.style.fill = colours.ink;
      // That same stylesheet strokes text inside a node. At eleven pixels the
      // stroke is wider than the glyph and paints the letter out entirely.
      letter.style.stroke = "none";
      letter.style.fontSize = "11px";
      letter.style.fontWeight = "700";
      letter.textContent = key.charAt(0).toUpperCase();
      badge.appendChild(letter);
    }}

    node.appendChild(badge);
  }}
}}

window.__sfDiagramsReady = (async () => {{
  const blocks = Array.prototype.slice.call(document.querySelectorAll("pre.mermaid"));
  if (blocks.length === 0) return 0;

  const namespace = window.__esbuild_esm_mermaid_nm;
  const mermaid = namespace && (namespace.mermaid.default || namespace.mermaid);
  if (!mermaid) return 0;

  mermaid.initialize({{
    startOnLoad: false,
    securityLevel: "strict",
    theme: "base",
    themeVariables: {{
      background: "transparent",
      primaryColor: {json.dumps(SURFACE)},
      primaryTextColor: {json.dumps(INK)},
      primaryBorderColor: {json.dumps(LINE)},
      lineColor: {json.dumps(LINE)},
      secondaryColor: {json.dumps(PAPER)},
      tertiaryColor: {json.dumps(PAPER)},
    }},
  }});

  let drawn = 0;
  for (let index = 0; index < blocks.length; index += 1) {{
    const block = blocks[index];
    const source = block.textContent || "";
    try {{
      const result = await mermaid.render("pdf-diagram-" + index, source);
      const figure = document.createElement("figure");
      figure.className = "diagram";
      figure.innerHTML = result.svg;
      const svg = figure.querySelector("svg");
      if (svg) paintBadges(svg, source);
      block.replaceWith(figure);
      drawn += 1;
    }} catch (error) {{
      // The source stays exactly where it was. A diagram that will not parse
      // costs the picture, never the document.
      console.error("diagram failed", error);
    }}
  }}
  return drawn;
}})();
"""
