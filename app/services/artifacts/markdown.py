"""Markdown primitives shared by every generator.

Small on purpose. What it buys is that a table rendered in the cost report and
the same table rendered in the whole-result document have identical column
handling — including the pipe-escaping, which is the one that bites. A tool
name containing `|` splits a row into an extra column, and the table renders as
garbage two documents downstream of the component that allowed it.
"""

from __future__ import annotations

from typing import Any

#: Markdown has no cell escape. A literal pipe has to become something else,
#: and a forward slash is the substitution that reads closest to the original.
PIPE_SUBSTITUTE = "/"


def scalar(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, list | tuple):
        return ", ".join(str(item) for item in value) or "—"
    if isinstance(value, dict):
        return ", ".join(f"{key}: {inner}" for key, inner in value.items()) or "—"
    return str(value)


def cell(value: Any) -> str:
    return scalar(value).replace("|", PIPE_SUBSTITUTE).replace("\n", " ")


def humanise(key: str) -> str:
    return key.replace("_", " ").replace("-", " ").capitalize()


def table(rows: list[dict[str, Any]], *, columns: list[str] | None = None) -> str:
    """A Markdown table from a list of row dicts.

    Columns come from the first row unless given. Rows missing a column get an
    empty cell rather than a shifted row — a ragged table that still parses is
    much easier to notice than one that silently misaligns.
    """
    if not rows:
        return ""
    keys = columns or list(rows[0].keys())
    header = " | ".join(humanise(key) for key in keys)
    divider = " | ".join("---" for _ in keys)
    body = "\n".join(
        "| " + " | ".join(cell(row.get(key, "")) for key in keys) + " |" for row in rows
    )
    return f"| {header} |\n| {divider} |\n{body}"


def key_values(values: dict[str, Any], *, key_header: str = "Key") -> str:
    if not values:
        return ""
    lines = "\n".join(f"| {humanise(key)} | {cell(value)} |" for key, value in values.items())
    return f"| {key_header} | Value |\n| --- | --- |\n{lines}"


def fence(content: str, *, language: str | None = None) -> str:
    """A fenced block whose fence is longer than any run of backticks inside it.

    A generated file containing a fenced Markdown block — the architecture
    document does, it embeds Mermaid — would otherwise close the outer fence
    early and dump the rest of the document as prose.
    """
    longest = 0
    run = 0
    for character in content:
        run = run + 1 if character == "`" else 0
        longest = max(longest, run)
    fence_marks = "`" * max(3, longest + 1)
    return f"{fence_marks}{language or ''}\n{content.rstrip()}\n{fence_marks}"
