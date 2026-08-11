"""The template file format (M19).

The claim under test is the definition-of-done line: *adding a template is a
Markdown file plus a seed run — no code change*. It is asserted by writing a
file to a temporary directory and loading it, because a registry listing every
slug would look tidier and would quietly make that claim false the first time
somebody forgot to edit it.

The rest is validation. There are thirty of these maintained by hand, so the
failure worth designing against is a silent one — a frontmatter typo that seeds
successfully with an empty title and is discovered on a live page.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.data.templates_loader import (
    CATEGORIES,
    TEMPLATES_DIR,
    TemplateError,
    load_all,
    load_file,
)

MINIMAL = """---
title: A Test Template
category: checklist
summary: One line about it.
---

# Body

Some prose.
"""


def write(directory: Path, name: str, body: str) -> Path:
    path = directory / name
    path.write_text(body, encoding="utf-8")
    return path


# ── the no-code-change claim ─────────────────────────────────────────────────


def test_a_new_markdown_file_is_found_with_no_code_change(tmp_path: Path) -> None:
    write(tmp_path, "brand-new.md", MINIMAL)

    seeds = load_all(tmp_path)

    assert [seed.slug for seed in seeds] == ["brand-new"]
    assert seeds[0].title == "A Test Template"


def test_the_slug_defaults_to_the_filename(tmp_path: Path) -> None:
    """So the URL and the file cannot drift apart without someone meaning it."""
    write(tmp_path, "some-slug.md", MINIMAL)
    assert load_all(tmp_path)[0].slug == "some-slug"


def test_templates_load_in_a_deterministic_order(tmp_path: Path) -> None:
    for name in ("zebra.md", "apple.md", "mango.md"):
        write(tmp_path, name, MINIMAL)

    assert [seed.slug for seed in load_all(tmp_path)] == ["apple", "mango", "zebra"]


def test_a_missing_directory_is_empty_rather_than_an_error(tmp_path: Path) -> None:
    assert load_all(tmp_path / "nothing-here") == []


# ── validation ───────────────────────────────────────────────────────────────


def test_a_file_with_no_frontmatter_is_refused(tmp_path: Path) -> None:
    path = write(tmp_path, "bare.md", "# Just a heading\n")
    with pytest.raises(TemplateError, match="frontmatter"):
        load_file(path)


def test_a_missing_required_field_names_the_field_and_the_file(tmp_path: Path) -> None:
    path = write(tmp_path, "untitled.md", "---\ncategory: checklist\nsummary: x\n---\n\nBody\n")
    with pytest.raises(TemplateError, match=re.escape("untitled.md: `title` is required")):
        load_file(path)


def test_an_unknown_category_is_refused(tmp_path: Path) -> None:
    path = write(tmp_path, "odd.md", MINIMAL.replace("checklist", "recipes"))
    with pytest.raises(TemplateError, match="category 'recipes'"):
        load_file(path)


def test_two_files_claiming_one_slug_are_refused(tmp_path: Path) -> None:
    """One would silently win the upsert, and which one depends on sort order."""
    write(tmp_path, "first.md", MINIMAL)
    write(tmp_path, "second.md", MINIMAL.replace("---\n\n# Body", "slug: first\n---\n\n# Body"))

    with pytest.raises(TemplateError, match="Duplicate template slug"):
        load_all(tmp_path)


def test_a_stack_template_without_a_payload_is_refused(tmp_path: Path) -> None:
    """The whole point of the category is that opening it loads the Architect
    form. One that cannot do that is mis-filed."""
    path = write(tmp_path, "not-really.md", MINIMAL.replace("checklist", "stack"))
    with pytest.raises(TemplateError, match="stack_input"):
        load_file(path)


def test_a_code_starter_without_files_is_refused(tmp_path: Path) -> None:
    path = write(tmp_path, "no-code.md", MINIMAL.replace("checklist", "code-starter"))
    with pytest.raises(TemplateError, match="fenced file"):
        load_file(path)


# ── the file fence ───────────────────────────────────────────────────────────


WITH_FILES = """---
title: A Starter
category: code-starter
summary: Two files.
---

Some prose before.

```python path=app/main.py
print("hello")
```

Some prose between.

```yaml path=config/settings.yaml
key: value
```
"""


def test_path_tagged_fences_become_files(tmp_path: Path) -> None:
    seed = load_file(write(tmp_path, "starter.md", WITH_FILES))

    assert [(file.path, file.language) for file in seed.files] == [
        ("app/main.py", "python"),
        ("config/settings.yaml", "yaml"),
    ]
    assert seed.files[0].content.strip() == 'print("hello")'


def test_a_fenced_file_is_removed_from_the_prose(tmp_path: Path) -> None:
    """A multi-file starter renders as a file tree with per-file copy. Leaving
    the content in the body as well would show every file twice — once
    copy-pastable and once not."""
    seed = load_file(write(tmp_path, "starter.md", WITH_FILES))

    assert "Some prose before." in seed.content_markdown
    assert "Some prose between." in seed.content_markdown
    assert 'print("hello")' not in seed.content_markdown
    # And the removal does not leave a three-line gap behind.
    assert "\n\n\n" not in seed.content_markdown


def test_an_untagged_fence_stays_in_the_prose(tmp_path: Path) -> None:
    """A shell snippet showing how to run the thing is documentation, not a
    file in the tree."""
    body = MINIMAL.replace("Some prose.", "```bash\nuv sync\n```")
    seed = load_file(write(tmp_path, "docs.md", body))

    assert seed.files == []
    assert "uv sync" in seed.content_markdown


def test_the_language_is_inferred_from_the_extension_when_omitted(tmp_path: Path) -> None:
    body = """---
title: Inferred
category: code-starter
summary: x
---

``` path=Dockerfile
FROM python:3.13-slim
```

``` path=.env.example
KEY=change-me
```
"""
    seed = load_file(write(tmp_path, "inferred.md", body))
    assert [file.language for file in seed.files] == ["dockerfile", "bash"]


# ── the shipped library ──────────────────────────────────────────────────────


def test_the_shipped_library_loads_and_is_complete() -> None:
    """`PRD.md` §15: thirty templates across seven categories at launch."""
    from collections import Counter

    seeds = load_all(TEMPLATES_DIR)
    by_category = Counter(seed.category for seed in seeds)

    assert len(seeds) == 30
    assert dict(by_category) == {
        "stack": 5,
        "blueprint": 5,
        "code-starter": 4,
        "prompt": 5,
        "config": 4,
        "checklist": 4,
        "business": 3,
    }
    assert set(by_category) == CATEGORIES


def test_every_shipped_template_has_a_usable_summary() -> None:
    """The summary is the card, the search result, and the meta description.
    An empty or one-word one is a page nobody clicks."""
    for seed in load_all(TEMPLATES_DIR):
        assert len(seed.summary) > 40, seed.slug
        assert len(seed.summary) < 400, seed.slug


def test_every_stack_template_carries_a_full_architect_payload() -> None:
    """Asserted against `RecommendIn` itself rather than against a list of key
    names — a payload that merely *looks* like the form is one that fails at
    the moment the user clicks 'Use this stack'."""
    from app.schemas.architect import RecommendIn

    stacks = [seed for seed in load_all(TEMPLATES_DIR) if seed.category == "stack"]
    assert len(stacks) == 5

    for seed in stacks:
        payload = RecommendIn.model_validate(seed.stack_input)
        # Defaults would validate too, so check the template actually chose.
        assert payload.model_dump(exclude_unset=True).keys() >= {
            "use_case",
            "scale_target",
            "monthly_budget",
            "sensitivity",
        }, seed.slug


def test_every_related_tool_slug_exists() -> None:
    """A dead internal link on the product's main SEO surface is worse than no
    link: it wastes the crawl and it 404s a reader mid-decision."""
    from app.data.tools_seed import TOOLS

    known = {
        "llm-pricing",
        "token-calculator",
        "embedding-cost",
        "budget-estimator",
        "compare-models",
        "compare-vector-db",
        "compare-stacks",
        "build-vs-buy",
        "chunk-estimate",
        "chunking-strategy",
        "pdf-tokens",
        "vectordb-estimate",
        "pipeline-cost",
        "architecture",
        "mcp-config",
        "function-schema",
        "rate-limits",
        "agent-cost",
        "workflow-plan",
        "vram-estimate",
        "gpu-cost",
        "cloud-cost",
        "k8s-estimate",
        "docker-compose",
        "readiness-checklist",
        "hours-saved",
        "model-roi",
        "implementation-cost",
        "compatibility",
        "graveyard",
    } | {tool.slug for tool in TOOLS}

    for seed in load_all(TEMPLATES_DIR):
        for slug in seed.related_tools:
            assert slug in known, f"{seed.slug} links to unknown tool '{slug}'"
