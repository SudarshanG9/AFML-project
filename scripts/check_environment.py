"""Print backend status without assuming CUDA / MPS is available."""
from avfuse.member3_runtime.provenance import environment_info

if __name__ == "__main__":
    for key, value in environment_info().items():
        print(f"{key}: {value}")
