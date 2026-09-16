"""``python -m saip`` forwards to the command-line entry point."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
