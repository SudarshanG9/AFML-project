"""``python -m avfuse.member3_runtime`` -> the Member 3 CLI."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
