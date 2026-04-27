"""Render a side-by-side comparison plot for one StreamTrack alternative.

Two panels (fardal15 only):
  1. Aligned (phi_1, phi_2) view with sample cloud and the alt's leading +
     trailing tracks overlaid; the streamspray-track reference is drawn as
     dashed lines on top of the alt for visual comparison.
  2. Cartesian-position difference ||alt_xyz(tp) - main_xyz(tp)|| as a
     function of tp, separately for leading and trailing arms.

Workflow (uncommitted helper; per-branch comparisons need the reference
track to be saved first):

    git checkout streamspray-track
    python compare_alternatives.py streamspray-track   # saves reference
    for br in alt-gcv alt-no-binning alt-rotating-frame alt-6d-closest \
              alt-strip-time-affine alt-auto-timerange alt-kdtree; do
        git checkout "$br"
        python compare_alternatives.py "$br"
    done
    git checkout streamspray-track
"""

import os
import sys

import numpy
from astropy import units as u
from matplotlib import pyplot

from galpy.df import fardal15spraydf
from galpy.orbit import Orbit
from galpy.potential import LogarithmicHaloPotential
from galpy.util import _rotate_to_arbitrary_vector, coords

REF = "streamspray-track"
REF_NPZ = "alt_reference_track.npz"
REF_NPZ_PAL5 = "alt_reference_track_pal5.npz"


def _align_rot(prog):
    """Build the sky-rotation matrix that aligns the progenitor's helio-
    centric angular momentum with +z (stream lies along phi_2 ~ 0).

    The angular momentum is computed in galpy's galactocentric Cartesian
    frame, then converted to the equatorial (ICRS) frame before building
    the rotation — ``radec_to_custom`` operates on equatorial unit
    vectors, so the rotation pole must be expressed in that frame."""
    sun = Orbit([0, 0, 0, 0, 0, 0], radec=True)
    sun.vxvv[0][1] = sun.vxvv[0][2] = sun.vxvv[0][4] = 0.0
    sun.turn_physical_off()
    p = prog()
    p.turn_physical_off()
    dx = p.x() - sun.x()
    dy = p.y() - sun.y()
    dz = p.z() - sun.z()
    dvx = p.vx() - sun.vx()
    dvy = p.vy() - sun.vy()
    dvz = p.vz() - sun.vz()
    Lx = dy * dvz - dz * dvy
    Ly = dz * dvx - dx * dvz
    Lz = dx * dvy - dy * dvx
    # Convert L direction from galpy galactocentric Cartesian to
    # heliocentric Galactic (l, b): galpy x points GC→Sun while
    # Galactic X points Sun→GC, so flip x.
    l_pole = numpy.degrees(numpy.arctan2(Ly, -Lx))
    b_pole = numpy.degrees(numpy.arctan2(Lz, numpy.sqrt(Lx**2 + Ly**2)))
    radec = numpy.asarray(coords.lb_to_radec(l_pole, b_pole, degree=True))
    ra_pole = float(radec[0, 0]) if radec.ndim == 2 else float(radec[0])
    dec_pole = float(radec[0, 1]) if radec.ndim == 2 else float(radec[1])
    L_eq = numpy.array(
        [
            numpy.cos(numpy.radians(dec_pole)) * numpy.cos(numpy.radians(ra_pole)),
            numpy.cos(numpy.radians(dec_pole)) * numpy.sin(numpy.radians(ra_pole)),
            numpy.sin(numpy.radians(dec_pole)),
        ]
    )
    return _rotate_to_arbitrary_vector(numpy.atleast_2d(L_eq), [0.0, 0.0, 1.0])[0]


def build_setup():
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    prog = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596],
        ro=8.0,
        vo=220.0,
    )
    spdf = fardal15spraydf(
        progenitor_mass=10**4.0 * u.Msun,
        progenitor=prog,
        pot=lp,
        tdisrupt=4.5 * u.Gyr,
        tail="both",
    )
    return spdf, _align_rot(prog)


def build_setup_warm():
    """Bovy14 orbit but with 1000x larger progenitor (warm/fat stream)."""
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    prog = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596],
        ro=8.0,
        vo=220.0,
    )
    spdf = fardal15spraydf(
        progenitor_mass=10**7.0 * u.Msun,
        progenitor=prog,
        pot=lp,
        tdisrupt=4.5 * u.Gyr,
        tail="both",
    )
    return spdf, _align_rot(prog)


def build_setup_pal5():
    from galpy.potential import MWPotential2014

    prog = Orbit.from_name("Pal 5")
    spdf = fardal15spraydf(
        progenitor_mass=2e4 * u.Msun,
        progenitor=prog,
        pot=MWPotential2014,
        tdisrupt=5.0 * u.Gyr,
        tail="both",
    )
    return spdf, _align_rot(prog)


def to_phi12(ra, dec, rot):
    ra = numpy.asarray(getattr(ra, "value", ra))
    dec = numpy.asarray(getattr(dec, "value", dec))
    # T=rot (not rot.T): rot maps equatorial L to z, so T=rot puts the
    # orbital plane at phi2=0 when L was computed in equatorial Cartesian.
    phi12 = coords.radec_to_custom(ra, dec, T=rot, degree=True)
    phi1 = phi12[:, 0]
    phi2 = phi12[:, 1]
    # Unwrap phi1 around the median so the stream doesn't split across the
    # -180/+180 boundary of radec_to_custom.
    med = numpy.median(phi1)
    phi1 = (phi1 - med + 180.0) % 360.0 - 180.0 + med
    return phi1, phi2


def arm_xyz_on_grid(arm, tp_grid):
    """Track xyz at a fixed tp_grid (clipped if outside arm's own range)."""
    return numpy.column_stack([arm.x(tp_grid), arm.y(tp_grid), arm.z(tp_grid)])


def _render(label, spdf, rot, ref_npz, out_png, stream_name, seed=4):
    """Render one branch's comparison plot (aligned view + diff vs reference)."""
    numpy.random.seed(seed)
    track = spdf.streamTrack(n=2000)
    samples = spdf.sample(n=2000)

    tp_lead = track.leading.tp_grid()
    tp_trail = track.trailing.tp_grid()
    xyz_lead = arm_xyz_on_grid(track.leading, tp_lead)
    xyz_trail = arm_xyz_on_grid(track.trailing, tp_trail)

    # Save ra/dec along track for aligned-view overlay
    ra_lead = numpy.asarray(
        getattr(track.leading.ra(tp_lead), "value", track.leading.ra(tp_lead))
    )
    dec_lead = numpy.asarray(
        getattr(track.leading.dec(tp_lead), "value", track.leading.dec(tp_lead))
    )
    ra_trail = numpy.asarray(
        getattr(track.trailing.ra(tp_trail), "value", track.trailing.ra(tp_trail))
    )
    dec_trail = numpy.asarray(
        getattr(track.trailing.dec(tp_trail), "value", track.trailing.dec(tp_trail))
    )

    if label == REF:
        numpy.savez(
            ref_npz,
            tp_lead=tp_lead,
            tp_trail=tp_trail,
            xyz_lead=xyz_lead,
            xyz_trail=xyz_trail,
            ra_lead=ra_lead,
            dec_lead=dec_lead,
            ra_trail=ra_trail,
            dec_trail=dec_trail,
        )
        print(f"  saved reference to {ref_npz}")

    have_ref = label != REF and os.path.exists(ref_npz)
    if have_ref:
        ref = numpy.load(ref_npz)

    fig, axes = pyplot.subplots(2, 1, figsize=(13, 7))

    # Panel 1: aligned (phi_1, phi_2)
    ax = axes[0]
    phi1_s, phi2_s = to_phi12(samples.ra(), samples.dec(), rot)
    ax.scatter(phi1_s, phi2_s, s=2, alpha=0.3, color="0.6", label="samples")
    for arm, col, aname in [
        (track.leading, "C3", "lead"),
        (track.trailing, "C0", "trail"),
    ]:
        p1, p2 = to_phi12(arm.ra(arm.tp_grid()), arm.dec(arm.tp_grid()), rot)
        ax.plot(p1, p2, col, lw=1.5, label=f"{label} {aname}")
    if have_ref:
        for ra_key, dec_key, is_first in [
            ("ra_lead", "dec_lead", True),
            ("ra_trail", "dec_trail", False),
        ]:
            p1, p2 = to_phi12(ref[ra_key], ref[dec_key], rot)
            ax.plot(
                p1,
                p2,
                "k--",
                lw=1.0,
                alpha=0.7,
                label=f"{REF} (ref)" if is_first else None,
            )
    # Auto-scale ylim from the sample phi_2 range with generous padding;
    # positive phi_2 up, streams curve downward to the sides.
    phi2_med = numpy.median(phi2_s)
    phi2_span = max(numpy.percentile(phi2_s, 99) - numpy.percentile(phi2_s, 1), 4.0)
    ax.set_ylim(phi2_med - 1.5 * phi2_span, phi2_med + 1.5 * phi2_span)
    ax.set_xlabel(r"$\phi_1$ [deg]")
    ax.set_ylabel(r"$\phi_2$ [deg]")
    ax.set_title(f"{stream_name} — {label}")
    ax.legend(loc="upper right", fontsize=8)

    # Panel 2: ||x_alt − x_ref|| vs tp
    ax = axes[1]
    if have_ref:
        for arm, ref_tp_key, ref_xyz_key, col in [
            (track.leading, "tp_lead", "xyz_lead", "C3"),
            (track.trailing, "tp_trail", "xyz_trail", "C0"),
        ]:
            ref_tp, ref_xyz = ref[ref_tp_key], ref[ref_xyz_key]
            tp_lo = max(min(ref_tp), min(arm.tp_grid()))
            tp_hi = min(max(ref_tp), max(arm.tp_grid()))
            tps = numpy.linspace(tp_lo, tp_hi, 500)
            d = numpy.linalg.norm(
                arm_xyz_on_grid(arm, tps)
                - numpy.column_stack(
                    [numpy.interp(tps, ref_tp, ref_xyz[:, j]) for j in range(3)]
                ),
                axis=1,
            )
            ax.plot(tps, d, col, label=ref_tp_key.replace("tp_", ""))
        ax.set_xlabel(r"$t_p$ [galpy units]")
        ax.set_ylabel(r"$\|x_{\rm alt} - x_{\rm main}\|$ [galpy]")
        ax.set_title(f"3D position difference vs {REF}")
        ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            "(reference — no diff)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    pyplot.tight_layout()
    pyplot.savefig(out_png, dpi=100)
    print(f"  wrote {out_png}")
    pyplot.close(fig)


REF_NPZ_WARM = "alt_reference_track_warm.npz"


def main(label):
    # Bovy14 stream (LogarithmicHaloPotential, q=0.9, tdisrupt=4.5 Gyr)
    spdf, rot = build_setup()
    print(f"[bovy14] {label}")
    _render(label, spdf, rot, REF_NPZ, f"alt_{label}.png", "fardal15 (Bovy14)")

    # Pal 5 stream (MWPotential2014, tdisrupt=5 Gyr)
    spdf_p5, rot_p5 = build_setup_pal5()
    print(f"[pal5]   {label}")
    _render(
        label,
        spdf_p5,
        rot_p5,
        REF_NPZ_PAL5,
        f"alt_pal5_{label}.png",
        "fardal15 (Pal 5)",
        seed=42,
    )

    # Warm stream (Bovy14 orbit, 10^7 Msun progenitor)
    spdf_w, rot_w = build_setup_warm()
    print(f"[warm]   {label}")
    _render(
        label,
        spdf_w,
        rot_w,
        REF_NPZ_WARM,
        f"alt_warm_{label}.png",
        "fardal15 (warm, 1e7 Msun)",
        seed=7,
    )


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else REF
    main(label)
