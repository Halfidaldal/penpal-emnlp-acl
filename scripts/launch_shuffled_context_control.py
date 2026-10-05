#!/usr/bin/env python
"""Run shuffled_context_control.py as a Hugging Face Job, or fetch its results.

    export HF_TOKEN=hf_...
    python scripts/launch_shuffled_context_control.py --repo <user>/penpal-shuffled-control
    python scripts/launch_shuffled_context_control.py --repo <user>/penpal-shuffled-control --download

The input table is uploaded to a private dataset repo, which also receives
the results.
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, run_uv_job, snapshot_download

JOB_SCRIPT = Path(__file__).resolve().parent / "shuffled_context_control.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="HF dataset repo for input and results")
    parser.add_argument("--input", default="data/human-ai/interim/interaction_level_stories_filtered.csv")
    parser.add_argument("--model", default="google/gemma-4-12B")
    parser.add_argument("--flavor", default="a100-large")
    parser.add_argument("--timeout", default="2h")
    parser.add_argument("--outdir", default="data/human-ai/processed")
    parser.add_argument("--download", action="store_true", help="download finished results to --outdir")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set")

    if args.download:
        snapshot_download(args.repo, repo_type="dataset", local_dir=args.outdir, token=token,
                          ignore_patterns=["input.csv", ".git*"])
        return

    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=args.input, path_in_repo="input.csv", repo_id=args.repo, repo_type="dataset")

    job = run_uv_job(
        script=str(JOB_SCRIPT),
        flavor=args.flavor,
        timeout=args.timeout,
        secrets={"HF_TOKEN": token},
        env={"TARGET_HF_REPO": args.repo, "MODEL_NAME": args.model},
    )
    print(f"Submitted job {getattr(job, 'id', job)}")


if __name__ == "__main__":
    main()
