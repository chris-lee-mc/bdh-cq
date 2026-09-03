"""CLI: python tools/aggregate_results.py results/ --out reports/<date>/

Walks a results/ directory and writes all_runs.csv, summary.csv, flags.csv,
and the six plots of FRAMEWORK_SPEC section 10 to --out.
"""

from __future__ import annotations

import argparse

from bdhx.results.aggregate import aggregate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", help="directory containing one subdir per run")
    parser.add_argument("--out", required=True, help="output directory for tables and plots")
    args = parser.parse_args(argv)

    out = aggregate(args.results_root, args.out)
    n_plots = len(out["plots"])
    print(
        f"wrote {out['all_runs']}, {out['summary']}, {out['flags']}, {n_plots} plot(s) to {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
