"""Small operational CLI.

uv run python -m app.cli generate-keypair
"""

from __future__ import annotations

import sys


def _generate_keypair() -> None:
    from app.services.token_service import generate_keypair

    private_pem, public_pem = generate_keypair()
    # Written as single-line escaped values so they paste straight into .env.
    print("AUTH_PRIVATE_KEY=" + private_pem.strip().replace("\n", "\\n"))
    print("AUTH_PUBLIC_KEY=" + public_pem.strip().replace("\n", "\\n"))


def _openapi() -> None:
    """Dump the schema without starting a server.

    CI generates `types/api.ts` from this rather than booting the API, so the
    drift check needs no database and no port.
    """
    import json

    from app.main import app

    print(json.dumps(app.openapi(), indent=2))


COMMANDS = {"generate-keypair": _generate_keypair, "openapi": _openapi}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("usage: python -m app.cli <command>", file=sys.stderr)
        print("commands: " + ", ".join(COMMANDS), file=sys.stderr)
        return 2
    COMMANDS[sys.argv[1]]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
