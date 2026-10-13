"""List, run or summarize the CAGA experiment protocol."""
from __future__ import annotations

import argparse
from pathlib import Path
import os

from journal_protocol import (
    PROTOCOL, make_runs, is_complete, run_one, window_table,
    summarize, paired_vs_baseline, caga_routing,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--tiers', nargs='+', type=int, choices=[1, 2, 3, 4, 5], default=[1])
    parser.add_argument('--seeds', nargs='+', type=int, default=list(PROTOCOL['seeds']))
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--execute', action='store_true', help='Run the listed experiments on a GPU.')
    action.add_argument('--summarize', action='store_true', help='Summarize completed runs.')
    parser.add_argument('--perf', action='store_true', help='Enable optional performance optimizations.')
    args = parser.parse_args()
    results = Path(args.results_dir).resolve()
    os.chdir(Path(__file__).resolve().parents[1])

    if args.summarize:
        table = window_table(str(results))
        if table.empty:
            print('No completed runs found.')
            return
        results.mkdir(parents=True, exist_ok=True)
        table.to_csv(results / 'window_metrics.csv', index=False)
        summarize(table).to_csv(results / 'summary.csv', index=False)
        paired_vs_baseline(table).to_csv(results / 'paired_comparisons.csv', index=False)
        caga_routing(str(results)).to_csv(results / 'routing.csv')
        print(f'Summaries written to {results}')
        return

    runs = make_runs(seeds=args.seeds, tiers=sorted(set(args.tiers)))
    for run in runs:
        complete = is_complete(str(results / run['name']), PROTOCOL['num_rounds'])
        print(f"{run['name']}: {'complete' if complete else 'pending'}", flush=True)
        if args.execute and not complete:
            status, seconds = run_one(run, str(results), use_perf=args.perf)
            if status != 0:
                raise SystemExit(f"{run['name']} failed (exit {status}); see its run.log.")
            if not is_complete(str(results / run['name']), PROTOCOL['num_rounds']):
                raise SystemExit(f"{run['name']} exited without completing all rounds; see its run.log.")
            print(f'Completed in {seconds / 60:.1f} minutes.', flush=True)
    if not args.execute:
        print('Plan only. Add --execute to start training.')


if __name__ == '__main__':
    main()
