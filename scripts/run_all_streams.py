"""Run all decent galstreams streams in parallel, saving figures and
pickled results. Not committed."""

import os
import pickle
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

RESULTS_DIR = os.environ.get(
    "GALSTREAM_RESULTS", "/tmp/galstream_stress_results"
)
os.makedirs(RESULTS_DIR, exist_ok=True)


def _worker(name):
    """Run one stream. Executed in a separate process — must import inside
    so each worker has its own galpy/matplotlib state."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        from stress_test_helpers import run_stream
        smoothing = os.environ.get("GALSTREAM_SMOOTHING") or None
        smoothing_factor = float(os.environ.get("GALSTREAM_SMOOTHING_FACTOR", "1.0"))
        n_particles = int(os.environ.get("GALSTREAM_N", "500"))
        result = run_stream(
            name,
            initial_mass_msun=4e4,
            tdisrupt_gyr=5.0,
            n=n_particles,
            max_refine_iters=3,
            match_length=True,
            show=True,
            smoothing=smoothing,
            smoothing_factor=smoothing_factor,
        )
        with open(f"{RESULTS_DIR}/{name}.pkl", "wb") as f:
            pickle.dump(result, f)
        return name, None
    except Exception as exc:
        tb = traceback.format_exc()
        return name, f"{type(exc).__name__}: {exc}\n{tb}"


def main(max_workers=24):
    # Build the stream list in the main process (cheaper than per worker)
    from stress_test_helpers import list_decent_streams
    names = list_decent_streams()
    # Ensure GD-1 is included and put first for quick feedback
    if "GD-1-I21" in names:
        names.remove("GD-1-I21")
    names.insert(0, "GD-1-I21")
    print(f"Running {len(names)} streams with {max_workers} workers")
    t0 = time.time()
    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_worker, n): n for n in names}
        done_count = 0
        for fut in as_completed(futs):
            name, err = fut.result()
            done_count += 1
            elapsed = time.time() - t0
            if err is None:
                print(f"[{done_count:3d}/{len(names)}] {elapsed:6.1f}s  {name}: OK")
            else:
                first_line = err.split("\n")[0]
                print(f"[{done_count:3d}/{len(names)}] {elapsed:6.1f}s  {name}: FAIL — {first_line}")
            results[name] = err
    total = time.time() - t0
    fails = [n for n, e in results.items() if e is not None]
    print(f"\nTotal time: {total:.1f}s ({total/60:.1f} min)")
    print(f"Succeeded: {len(names) - len(fails)} / {len(names)}")
    if fails:
        print(f"Failed streams: {fails}")


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    main(max_workers=workers)
