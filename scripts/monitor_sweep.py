#!/usr/bin/env python3
"""Progress + ETA for a band-map sweep (run ON the box where results/ lives).

Counts completed train-seeds (each writes filtering/trainseed_<seed>/summary.json),
estimates the per-seed wall time from those summaries' mtimes, and projects the ETA
for the current seed and for all of them.

Usage:
  python3 scripts/monitor_sweep.py results/hellaswag_bounds_epochs_0615 7
  # on brandy (sh):  micromamba run -n w2s-overlap python scripts/monitor_sweep.py <root> 7
"""
import glob
import os
import subprocess
import sys
import time


def hm(secs: float) -> str:
    secs = max(0, int(secs))
    return f"{secs // 3600}h{(secs % 3600) // 60:02d}m"


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: monitor_sweep.py <output_root> [total_seeds]")
        sys.exit(1)
    root = sys.argv[1].rstrip("/")
    total = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    now = time.time()

    seed_files = sorted(
        glob.glob(os.path.join(root, "filtering", "trainseed_*", "summary.json")),
        key=os.path.getmtime,
    )
    done = len(seed_files)
    seed_mtimes = [os.path.getmtime(f) for f in seed_files]

    base_files = glob.glob(os.path.join(root, "*baseline*", "summary.json"))
    base_m = max((os.path.getmtime(f) for f in base_files), default=None)

    # per-seed wall times = gaps between consecutive completion marks
    marks = ([base_m] if base_m else []) + seed_mtimes
    durs = [b - a for a, b in zip(marks, marks[1:]) if b > a]
    avg = sum(durs) / len(durs) if durs else None

    try:
        alive = int(subprocess.run(["pgrep", "-fc", "run_dream_paper_style_lora.py"],
                                   capture_output=True, text=True).stdout.strip() or "0")
    except Exception:
        alive = -1

    print(f"output      : {root}")
    print(f"alive procs : {alive if alive >= 0 else '?'}  ({'RUNNING' if alive else 'NOT running'})")
    print(f"seeds done  : {done}/{total}")
    if durs:
        print(f"per-seed    : avg {hm(avg)} (last {len(durs)}: " + ", ".join(hm(d) for d in durs[-5:]) + ")")

    if done >= total:
        print("STATUS      : ALL SEEDS DONE")
        return
    if avg is None:
        if base_m:
            print(f"STATUS      : baseline done {hm(now - base_m)} ago; seed #1 in progress (need 1 finished seed for ETA)")
        else:
            print("STATUS      : warming up (no baseline/seed summary yet) -- ETA unknown")
        return

    last = seed_mtimes[-1] if seed_mtimes else base_m
    into_cur = now - last
    cur_left = max(0.0, avg - into_cur)
    eta_cur = now + cur_left
    all_left = max(0.0, (total - done) * avg - into_cur)
    eta_all = now + all_left

    print(f"current seed: #{done + 1}  ({hm(into_cur)} in, ~{hm(cur_left)} left)  "
          f"ETA {time.strftime('%a %H:%M', time.localtime(eta_cur))}")
    print(f"all {total} seeds : ~{hm(all_left)} left  "
          f"ETA {time.strftime('%a %H:%M', time.localtime(eta_all))}")


if __name__ == "__main__":
    main()
