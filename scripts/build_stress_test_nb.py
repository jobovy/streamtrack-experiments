"""Generate galstream_stress_test.ipynb. Each stream cell just loads the
precomputed figure + pickle from /tmp/galstream_stress_{figs,results}/
produced by run_all_streams.py. Not committed to git.

Respects env vars ``GALSTREAM_FIGDIR``, ``GALSTREAM_RESULTS``, and
``GALSTREAM_NB`` to support a parallel DOF-smoothing run alongside the
original GCV run."""

import os
import galstreams
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(
    nbf.v4.new_markdown_cell(
        """# streamspraydf + streamTrack stress test on galstreams streams

Tests the new `streamTrack` implementation on **all decent galstreams
streams** (filtered: has pm, has D, width_phi2 ≥ 0.05°, length ≥ 3°).
For each stream a 5-Gyr `fardal15spraydf` is built in `MWPotential2014`,
starting at 4×10⁴ M☉ progenitor mass. Target width is
`0.3 × galstreams.width_phi2` (galstreams widths are STREAMFINDER search
half-widths, ~3–5× the intrinsic Gaussian σ). The progenitor mass is
rescaled with σ ∝ M^(1/3) until the width matches.

**Compute is done out-of-band** by `run_all_streams.py` (parallel across
cores). This notebook just displays the results."""
    )
)

setup_code = '''# Loads precomputed results. Run run_all_streams.py first to fill
# /tmp/galstream_stress_{figs,results}/.
import glob, os, pickle
from IPython.display import Image, display

FIGS = os.environ.get("GALSTREAM_FIGDIR", "/tmp/galstream_stress_figs")
RES = os.environ.get("GALSTREAM_RESULTS", "/tmp/galstream_stress_results")

def load_result(name):
    path = f"{RES}/{name}.pkl"
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def show_stream(name):
    r = load_result(name)
    if r is None:
        print(f"{name}: no pickled result found")
    else:
        print(f"{name}:  galstreams width = {r['galstreams_width']:.3f}°, "
              f"target = {r['target_width']:.3f}°, "
              f"baseline w = {r['baseline_width']:.3f}°, "
              f"rescaled w = {r['final_width']:.3f}°, "
              f"M = {r['final_mass']:.2e} M_sun")
    fig = f"{FIGS}/{name}.png"
    if os.path.exists(fig):
        display(Image(fig))
    else:
        print(f"  (figure {fig} missing)")
'''
cells.append(nbf.v4.new_code_cell(setup_code))

# Discover the stream list at notebook-build time
_mws_build = galstreams.MWStreams(verbose=False, implement_Off=False)
stream_names = []
for _name in _mws_build.summary.index:
    try:
        _tr = _mws_build[_name]
    except KeyError:
        continue
    _flags = _tr.InfoFlags
    _w = _tr.track_width.get("width_phi2")
    if _w is None:
        continue
    _wv = _w.to_value("deg")
    if _wv < 0.05:
        continue
    if _flags[2] != "1":
        continue
    if _flags[1] not in ("1", "2"):
        continue
    if _tr.length.to_value("deg") < 3.0:
        continue
    stream_names.append((_name, _tr.length.to_value("deg")))

stream_names.sort(key=lambda x: -x[1])
stream_names = [n for n, _ in stream_names]
if "GD-1-I21" in stream_names:
    stream_names.remove("GD-1-I21")
stream_names.insert(0, "GD-1-I21")

print(f"Notebook will display {len(stream_names)} streams "
      "(GD-1 first, rest sorted by length).")

cells.append(
    nbf.v4.new_markdown_cell(
        "## Summary table\n\n"
        "Table of every stream's galstreams width, scaled target, "
        "baseline and rescaled widths, and fitted mass."
    )
)
cells.append(
    nbf.v4.new_code_cell(
        """names = sorted(
    [os.path.basename(p)[:-4] for p in glob.glob(f"{RES}/*.pkl")],
    key=lambda n: -load_result(n)['galstreams_width'] * 0 + (
        0 if n == 'GD-1-I21' else 1
    )
)
# Keep GD-1 first, rest by galstreams width descending
if 'GD-1-I21' in names:
    names.remove('GD-1-I21')
names = ['GD-1-I21'] + sorted(names, key=lambda n: -load_result(n)['galstreams_width'])

print(f"{'stream':<22}  {'gs w':>6}  {'targ':>6}  {'Mfit w':>7}  "
      f"{'Mfit [M☉]':>9}  {'L_tgt':>6}  {'L_fit':>6}  {'T_fit [Gyr]':>11}")
for n in names:
    r = load_result(n)
    if r is None: continue
    L_tgt = r.get('target_length', float('nan'))
    L_fit = r.get('final_length', float('nan'))
    T_fit = r.get('final_tdisrupt_gyr', float('nan'))
    print(f"{r['name']:<22}  {r['galstreams_width']:>5.2f}°  "
          f"{r['target_width']:>5.2f}°  {r['final_width']:>6.2f}°  "
          f"{r['final_mass']:>9.2e}  {L_tgt:>5.0f}°  {L_fit:>5.0f}°  "
          f"{T_fit:>10.2f}")
"""
    )
)

for i, name in enumerate(stream_names):
    if i == 0:
        cells.append(nbf.v4.new_markdown_cell(
            f"## Stream {i}: `{name}` (well-studied reference)"
        ))
    else:
        cells.append(nbf.v4.new_markdown_cell(f"## Stream {i}: `{name}`"))
    cells.append(nbf.v4.new_code_cell(f"show_stream({name!r})"))

nb["cells"] = cells

out = os.environ.get(
    "GALSTREAM_NB", "/home/bovy/Repos/galpy/galstream_stress_test.ipynb"
)
with open(out, "w") as f:
    nbf.write(nb, f)

print(f"Wrote {out}")
