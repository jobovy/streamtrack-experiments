# StreamTrack experiments

Exploratory work-in-progress that fed into the `StreamTrack` /
`StreamTrackPair` machinery in **galpy PR #861**
([`jobovy/galpy#861`](https://github.com/jobovy/galpy/pull/861)). None of
this is needed to *use* `streamspraydf.streamTrack` from the merged PR —
it's the pile of alternative implementations, stress tests, and
comparison plots that informed the design choices.

The `streamspray-track` branch on the galpy repo is the version that
shipped; everything here is the **scratch directory** of experiments
around it.

## Directory layout

```
streamtrack-experiments/
├── README.md                         (this file)
├── STREAMTRACK_ALTERNATIVES.md       7 alternative fitter implementations
├── STREAM_ORBIT_COMPARISON.md        v1 vs v2 stream-orbit refinement
├── TRIAXIAL_COMPARISON.md            Triaxial NFW: main vs v1 vs v2
├── WARM_STREAM_RESULTS.md            Combined-algorithm warm/cold results
│
├── notebooks/
│   ├── stream_track_examples.ipynb           Bovy14 + Pal5 worked examples
│   ├── gd1_track_example.ipynb               GD-1 with custom_transform
│   └── galstream_stress_test.ipynb           ~30 streams from galstreams (~30 MB)
│
├── scripts/
│   ├── compare_alternatives.py               diff plots vs main reference
│   ├── run_all_streams.py                    drives the galstreams stress test
│   ├── stress_test_helpers.py                helpers for the stress notebook
│   ├── triaxial_plot.py                      renders TRIAXIAL_COMPARISON figures
│   ├── build_stream_track_examples.py        regenerates stream_track_examples.ipynb
│   ├── build_gd1_track_example.py            regenerates gd1_track_example.ipynb
│   └── build_stress_test_nb.py               regenerates galstream_stress_test.ipynb
│
├── results/
│   ├── alt_reference_track.npz               Bovy14 reference track (used by diff)
│   ├── alt_reference_track_pal5.npz          Pal 5 reference track
│   └── alt_reference_track_warm.npz          warm-stream reference
│
└── plots/
    ├── alternatives/                7 alt-branch comparison plots × 3 streams
    ├── triaxial/                    Triaxial-NFW comparison (cold/warm × main/v1/v2)
    ├── warm_stream/                 Warm vs cold stream tracks (with stream-orbit variants)
    ├── gd1_example/                 GD-1 notebook output
    ├── stream_track_examples/       stream_track_examples.ipynb output
    └── sec4_0.png                   Section-4 figure
```

## What each topic covers

### `STREAMTRACK_ALTERNATIVES.md` — fitter alternatives

Seven branches off `streamspray-track`, each replacing one piece of the
fitter. The shipped main branch uses **closest-point projection onto a
short-range progenitor orbit + bin-then-smooth offsets with
`UnivariateSpline`**; the alternatives swap that for GCV smoothing,
per-particle (binless) splines, smoothing in a progenitor-aligned
rotating frame, 6D closest-point, stripping-time-as-tp, adaptive
`track_time_range`, or a KD-tree closest-point. Plots compare each
alternative's track against the main reference for **Bovy14**, **Pal 5**,
and a warm-stream variant.

The headline diff readout (galpy internal units × 8 ≈ kpc) is in the
table at the bottom of `STREAMTRACK_ALTERNATIVES.md`. The final shipped
algorithm includes pieces from `alt-gcv` (auto-smoothing), `alt-6d-closest`
(disambiguation), `alt-auto-timerange` (data-driven track range), and
`alt-kdtree` (sublinear closest-point).

Plots live in `plots/alternatives/`. Filename convention: `alt_<stream>_<variant>.png`
where `<stream>` ∈ {`""` (Bovy14), `"pal5_"`, `"warm_"`} and `<variant>` is the
branch name (e.g., `alt-gcv`, `alt-kdtree`).

### `STREAM_ORBIT_COMPARISON.md` — stream-orbit refinement

Two attempts at replacing the progenitor orbit with a "stream orbit"
integrated from the stream tip:

* **v1** — one-shot orbit from the mean of the outer 10% of particles
  (noisy IC causes divergence on warm streams).
* **v2** — two-step: fit a preliminary track on the progenitor, evaluate
  it at the tip for a noise-free IC, then re-fit using that orbit as
  base.

v2 fixes v1's divergence but the improvement over the unrefined main
branch is modest for the streams tested. The work is on the
`alt-stream-orbit` branch but **was not promoted** to PR #861.

Plots in `plots/warm_stream/warm_test_stream_orbit_*.png`.

### `TRIAXIAL_COMPARISON.md` — main vs v1 vs v2 in a triaxial halo

`TriaxialNFWPotential(a=20/8, b=0.8, c=0.6)`, Bovy14-like progenitor,
2 Gyr disruption, n=1000. Same three implementations on cold (10⁴ M☉)
and warm (10⁷ M☉) streams in three projections each.

Plots in `plots/triaxial/triaxial_{main,v1,v2}_{cold,warm}_{1e4,1e7}.png`.

### `WARM_STREAM_RESULTS.md` — combined algorithm on warm + cold

The shipped fitter (GCV + 6D matching + auto-timerange + boundary
exclusion) on cold (10⁴ M☉) and warm (10⁷ M☉) Bovy14 streams, six
projections each.

Plots in `plots/warm_stream/warm_test_{cold,warm}_*.png`.

## Notebooks

* **`stream_track_examples.ipynb`** — the canonical end-to-end notebook.
  Bovy14 + Pal 5; covers basic usage, accessors, `cov(basis=...)`, custom
  transforms, smoothing reuse. The `streamspraydf` tutorial in galpy is
  a trimmed-down version of this.

* **`gd1_track_example.ipynb`** — GD-1-like stream with `custom_transform`
  set up via `Orbit.align_to_orbit()`. The plots in `plots/gd1_example/`
  are this notebook's outputs at three different progenitor variants
  (`gd1_*`, `gd1b_*`, `gd1c_*`).

* **`galstream_stress_test.ipynb`** — runs the fitter against ~30 known
  streams from the `galstreams` library to flush out edge cases. Large
  (~30 MB) because every stream has a multi-panel plot. Drives:
  `scripts/run_all_streams.py` + `scripts/stress_test_helpers.py`.

## Scripts

* **`compare_alternatives.py`** — given a branch name, samples the
  Bovy14 + Pal 5 streams, fits with the current `streamTrack`, plots
  it on top of the sample cloud, and (when comparing against the
  saved reference) plots the per-tp 3D position diff. Output:
  `alt_<branch>.png` and `alt_<stream>_<branch>.png`. Used to generate
  `plots/alternatives/`.

* **`run_all_streams.py` + `stress_test_helpers.py`** — drives the
  galstreams stress notebook. Loads each known stream, fits, makes a
  multi-panel plot.

* **`triaxial_plot.py`** — renders the three implementations on the
  triaxial-NFW test (`plots/triaxial/`).

* **`build_*.py`** — these are the notebook *generators*: each
  programmatically writes the corresponding `.ipynb` file in
  `notebooks/`. Re-run them after editing if you want to refresh
  outputs.

## Reproducing

These scripts assume `galpy` is installed and importable. The
alternative-branch comparison loop uses `git checkout alt-<name>` on the
`galpy` source — those branches were pushed to the galpy fork at the
time of writing but may not still exist there.

```bash
# in a galpy clone, on the streamspray-track branch
python streamtrack-experiments/scripts/compare_alternatives.py streamspray-track
```

## Status

This directory is exploratory and **not maintained alongside galpy**.
The merged-and-shipped version of the work is in
`galpy/df/streamTrack.py` on the main branch. If something in here
diverges from what shipped, the shipped code wins.
