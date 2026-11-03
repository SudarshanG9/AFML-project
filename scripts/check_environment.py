"""Print backend status without assuming CUDA / MPS is available."""
import sys
from pathlib import Path

# Ensure the project root is on sys.path so 'src' is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.avfuse.member3_runtime.provenance import environment_info

if __name__ == "__main__":
    for key, value in environment_info().items():
        print(f"{key}: {value}")
