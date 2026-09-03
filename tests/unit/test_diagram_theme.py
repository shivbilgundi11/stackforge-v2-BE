"""Diagram colour and brand metadata.

The property worth asserting is that the extra lines are *additive*: a renderer
that knows nothing about them draws exactly the diagram it drew before. Colour
is ordinary Mermaid and brand marks are comments, so the picture survives being
pasted anywhere, and that is the only reason the metadata is allowed to live in
the source at all.
"""

from __future__ import annotations

from app.data.brands import BRANDS, brand_for
from app.services import diagram_theme


def test_a_role_gets_its_own_colour_and_an_unknown_one_falls_back() -> None:
    assert diagram_theme.ROLE_COLOURS["llm"] != diagram_theme.ROLE_COLOURS["cache"]
    # An agent node named after the work it does rather than a stack role.
    assert "sf-mystery" in "\n".join(diagram_theme.classdef_block(["mystery"]))
    assert diagram_theme.DEFAULT_COLOUR in "\n".join(diagram_theme.classdef_block(["mystery"]))


def test_one_classdef_per_role_however_many_nodes_use_it() -> None:
    """A diagram carrying twenty definitions for four boxes is noise in a file
    somebody reads."""
    block = diagram_theme.classdef_block(["database", "database", "cache"])

    assert len(block) == 2


def test_the_stroke_is_set_and_the_fill_is_not() -> None:
    """A fixed fill cannot work in both places: light enough to read black text
    on is invisible against the dark app surface, and dark enough for the app
    is a hole in a printed page. The stroke carries the colour in both."""
    line = diagram_theme.classdef_block(["llm"])[0]

    assert "stroke:" in line
    assert "fill:" not in line


def test_a_brand_comment_names_the_icon_and_its_colour() -> None:
    comment = diagram_theme.brand_comment("database", "postgresql", "database")

    assert comment == f"%% brand:database:postgresql:{BRANDS['postgresql'].hex}"


def test_a_tool_with_no_icon_still_gets_the_roles_colour() -> None:
    """Half the catalog has no mark in the set these are drawn from. Those get
    a monogram, and it should be the role's colour rather than a grey circle."""
    assert brand_for("weaviate") is None

    comment = diagram_theme.brand_comment("vector_db", "weaviate", "vector_db")

    assert comment is not None
    assert comment.endswith(diagram_theme.ROLE_COLOURS["vector_db"].lstrip("#"))
    assert ":: " not in comment
    assert comment.startswith("%% brand:vector_db::")


def test_a_node_with_no_tool_gets_no_comment() -> None:
    # `client` is the caller, not a component. Nothing to brand.
    assert diagram_theme.brand_comment("client", None, "client") is None


def test_decorate_leaves_the_diagram_untouched_above_what_it_adds() -> None:
    body = ["graph LR", '    database["Database"]', "    client --> database"]

    out = diagram_theme.decorate(
        body, roles={"client": "client", "database": "database"}, tools={"database": "postgresql"}
    )

    assert out[: len(body)] == body
    assert "    class database sf-database" in out
    # Comments last, so a reader opening the file meets the picture first.
    assert out.index("    %% brand:database:postgresql:4169E1") > out.index(
        "    class database sf-database"
    )


def test_every_mapped_icon_has_a_six_digit_hex_without_a_hash() -> None:
    """The hex travels through a colon-separated comment, where a `#` would
    read as the start of a fragment to anything parsing a URL out of it."""
    for slug, mark in BRANDS.items():
        assert len(mark.hex) == 6, slug
        assert not mark.hex.startswith("#"), slug
        int(mark.hex, 16)
