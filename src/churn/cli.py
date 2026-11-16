"""Command line entry point:  churn simulate | churn run."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from churn.config import Settings
from churn.validate import DataValidationError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="churn", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    commands = parser.add_subparsers(dest="command", required=True)

    simulate = commands.add_parser("simulate", help="generate the simulated shop data")
    simulate.add_argument("--out", type=Path, default=Path("data/raw"))
    simulate.add_argument("--seed", type=int, default=42)
    simulate.add_argument("--customers", type=int, default=5000)

    run = commands.add_parser("run", help="run the full analysis and write reports/")
    run.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    run.add_argument("--report-dir", type=Path, default=Path("reports"))
    run.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        if args.command == "simulate":
            from churn.simulate import simulate

            simulate(args.out, args.seed, args.customers)
        else:
            from churn.pipeline import run

            run(Settings(data_dir=args.data_dir, report_dir=args.report_dir, seed=args.seed))
    except DataValidationError as error:
        logging.getLogger("churn").error("invalid input data: %s", error)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
