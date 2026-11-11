from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from badminton_decision.constants import DATASET_REPO_ID


def download_dataset(output_dir: str | Path, repo_id: str = DATASET_REPO_ID, revision: str = "main", token: str | None = None) -> Path:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(output_dir),
        token=token,
        allow_patterns=[
            "videos/**",
            "finebadminton-20K/**",
            "benchmark_source_videos.txt",
            "README.md",
        ],
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Finebadminton-20K from Hugging Face.")
    parser.add_argument("--output-dir", default="data/raw/finebadminton-20k")
    parser.add_argument("--repo-id", default=DATASET_REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--token", default=None, help="Optional Hugging Face token; env HF_TOKEN also works.")
    args = parser.parse_args()
    path = download_dataset(args.output_dir, args.repo_id, args.revision, args.token)
    print(f"Dataset snapshot is available at: {path}")


if __name__ == "__main__":
    main()
