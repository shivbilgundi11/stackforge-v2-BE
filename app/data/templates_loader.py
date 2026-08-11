"""Reading the template library off disk (M19).

Templates are Markdown files with YAML frontmatter under
`app/data/templates/<category>/<slug>.md`. This module turns that directory
into typed records; `seed_service` writes them to the database.

**The directory is the source of truth, and nothing here is registered in
code.** Adding a template is dropping a file in and running the seeder — the
definition of done says so, and the test that proves it writes a file to a
temporary directory and asserts the loader finds it. A registry listing every
slug would look tidier and would quietly make that claim false the first time
someone forgot to edit it.

Validation is strict and loud. A frontmatter typo produces a `TemplateError`
naming the file and the field, at load time, rather than a row that seeds
successfully with an empty title and is discovered on a live page. There are
thirty of these maintained by hand; the failure mode worth designing against
is a silent one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import yaml

#: `app/data/templates/`. Resolved from this file so it survives being run from
#: any working directory — the seeder is invoked from the repo root, from the
#: CLI, and from pytest, and all three have different ideas of `.`.
TEMPLATES_DIR: Final = Path(__file__).parent / "templates"

CATEGORIES: Final = frozenset(
    {"stack", "blueprint", "code-starter", "prompt", "config", "checklist", "business"}
)
DIFFICULTIES: Final = frozenset({"beginner", "intermediate", "advanced"})

#: `---\n<yaml>\n---\n<body>`. Anchored at the start: a document that merely
#: *contains* a horizontal rule is not a document with frontmatter.
FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)

#: A fenced block carrying a path becomes a file in a multi-file starter:
#:
#:     ```python path=app/main.py
#:
#: Chosen over a `files:` block in the frontmatter because a code starter is
#: read far more often than it is seeded, and YAML-escaped source code is
#: unreadable and unlintable. In a fence it is just code.
FILE_FENCE = re.compile(
    r"^```(?P<language>[\w+-]*)\s+path=(?P<path>[^\s`]+)\s*\n(?P<content>.*?)^```\s*$",
    re.DOTALL | re.MULTILINE,
)


class TemplateError(ValueError):
    """A template file that cannot be trusted. Names the file and the field."""


@dataclass(frozen=True)
class TemplateFile:
    path: str
    language: str
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"path": self.path, "language": self.language, "content": self.content}


@dataclass(frozen=True)
class TemplateSeed:
    slug: str
    title: str
    category: str
    difficulty: str
    summary: str
    content_markdown: str
    files: list[TemplateFile] = field(default_factory=list)
    stack_input: dict[str, Any] = field(default_factory=dict)
    use_cases: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    related_tools: list[str] = field(default_factory=list)
    is_premium: bool = False

    @property
    def is_multi_file(self) -> bool:
        return bool(self.files)


def load_all(directory: Path | None = None) -> list[TemplateSeed]:
    """Every template under `directory`, sorted by slug.

    Sorted so a seed run is deterministic and two runs produce the same insert
    order — which is what makes "seeding is idempotent" a claim about the data
    rather than about the filesystem's iteration order.
    """
    root = directory or TEMPLATES_DIR
    if not root.exists():
        return []

    seeds = [load_file(path) for path in sorted(root.rglob("*.md"))]

    duplicates = _duplicates([seed.slug for seed in seeds])
    if duplicates:
        # Two files claiming one slug means one silently wins the upsert, and
        # which one depends on sort order. Refusing is the only honest answer.
        raise TemplateError(f"Duplicate template slug(s): {', '.join(sorted(duplicates))}.")

    return sorted(seeds, key=lambda seed: seed.slug)


def _duplicates(values: list[str]) -> set[str]:
    seen: set[str] = set()
    repeated: set[str] = set()
    for value in values:
        if value in seen:
            repeated.add(value)
        seen.add(value)
    return repeated


def load_file(path: Path) -> TemplateSeed:
    raw = path.read_text(encoding="utf-8")
    match = FRONTMATTER.match(raw)
    if match is None:
        raise TemplateError(f"{path.name}: no YAML frontmatter block at the top of the file.")

    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as error:
        raise TemplateError(f"{path.name}: frontmatter is not valid YAML — {error}") from error
    if not isinstance(meta, dict):
        raise TemplateError(f"{path.name}: frontmatter must be a mapping.")

    body = match.group(2)
    files, content = _split_files(body)

    # The slug defaults to the filename, so the common case needs no field and
    # the URL and the file cannot drift apart.
    slug = str(meta.get("slug") or path.stem).strip()
    category = _required(meta, "category", path)
    difficulty = str(meta.get("difficulty") or "intermediate").strip()

    if category not in CATEGORIES:
        raise TemplateError(
            f"{path.name}: category '{category}' is not one of {', '.join(sorted(CATEGORIES))}."
        )
    if difficulty not in DIFFICULTIES:
        raise TemplateError(
            f"{path.name}: difficulty '{difficulty}' is not one of "
            f"{', '.join(sorted(DIFFICULTIES))}."
        )

    stack_input = meta.get("stack_input") or {}
    if not isinstance(stack_input, dict):
        raise TemplateError(f"{path.name}: stack_input must be a mapping.")
    if category == "stack" and not stack_input:
        # A stack template with no payload is a document pretending to be one.
        # The whole point of the category is that opening it loads the Stack
        # Architect form, and one that cannot do that is mis-filed.
        raise TemplateError(
            f"{path.name}: a stack template needs a `stack_input` block — it is what "
            f"the 'Use this stack' button loads into the Architect."
        )
    if category == "code-starter" and not files:
        raise TemplateError(
            f"{path.name}: a code starter needs at least one ```lang path=… fenced file."
        )

    return TemplateSeed(
        slug=slug,
        title=_required(meta, "title", path),
        category=category,
        difficulty=difficulty,
        summary=_required(meta, "summary", path),
        content_markdown=content.strip() + "\n",
        files=files,
        stack_input=stack_input,
        use_cases=_string_list(meta.get("use_cases"), "use_cases", path),
        tags=_string_list(meta.get("tags"), "tags", path),
        related_tools=_string_list(meta.get("related_tools"), "related_tools", path),
        is_premium=bool(meta.get("premium", False)),
    )


def _required(meta: dict[str, Any], key: str, path: Path) -> str:
    value = meta.get(key)
    if value is None or not str(value).strip():
        raise TemplateError(f"{path.name}: `{key}` is required in the frontmatter.")
    return str(value).strip()


def _string_list(value: Any, key: str, path: Path) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        # A single tag written unquoted is a string, not a one-item list.
        # Accepting it rather than raising is the right trade: this is a hand-
        # maintained format and the intent is unambiguous.
        return [value.strip()]
    if not isinstance(value, list):
        raise TemplateError(f"{path.name}: `{key}` must be a list of strings.")
    return [str(item).strip() for item in value if str(item).strip()]


def _split_files(body: str) -> tuple[list[TemplateFile], str]:
    """Pull `path=`-tagged fences out of the body into the file tree.

    The fence is *removed* from the prose rather than left in it. A multi-file
    starter renders as a file tree with per-file copy, and leaving the same
    content in the body as well would show every file twice — once
    copy-pastable and once not.
    """
    files: list[TemplateFile] = []

    def take(match: re.Match[str]) -> str:
        files.append(
            TemplateFile(
                path=match.group("path"),
                language=match.group("language") or _language_of(match.group("path")),
                content=match.group("content"),
            )
        )
        return ""

    remaining = FILE_FENCE.sub(take, body)
    # Collapse the blank runs the removed fences left behind, so the prose does
    # not render with three-line gaps where a file used to be.
    remaining = re.sub(r"\n{3,}", "\n\n", remaining)
    return files, remaining


EXTENSION_LANGUAGES: Final[dict[str, str]] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "jsx",
    ".json": "json",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
    ".sh": "bash",
    ".sql": "sql",
    ".env": "bash",
    ".txt": "text",
    ".dockerfile": "dockerfile",
}


def _language_of(path: str) -> str:
    name = Path(path).name.lower()
    if name.startswith(".env"):
        return "bash"
    if name in {"dockerfile", "containerfile"}:
        return "dockerfile"
    if name == "makefile":
        return "makefile"
    return EXTENSION_LANGUAGES.get(Path(path).suffix.lower(), "text")


__all__ = [
    "CATEGORIES",
    "DIFFICULTIES",
    "TEMPLATES_DIR",
    "TemplateError",
    "TemplateFile",
    "TemplateSeed",
    "load_all",
    "load_file",
]
