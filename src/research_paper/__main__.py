"""Build the paper's exhibits: ``uv run python -m src.research_paper [--stages ...] [--force]``."""

from __future__ import annotations

import argparse
import logging

from src.research_paper.pipeline import PaperPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", help=f"Comma-separated subset of: {','.join(PaperPipeline.STAGES)}. Default: all.")
    parser.add_argument("--force", action="store_true", help="Rescore every central bank and year instead of reusing cached scores.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    stages = args.stages.split(",") if args.stages else None
    for path in PaperPipeline(force=args.force).run(stages):
        print(path)


if __name__ == "__main__":
    main()
