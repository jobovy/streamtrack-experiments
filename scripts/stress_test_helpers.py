"""Helpers for the galstreams stress test — extracted so both the
notebook and a parallel driver can import them. Not committed."""

import warnings

warnings.filterwarnings("ignore")

import os
import time

import astropy.units as u
import galstreams
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord

from galpy.df import fardal15spraydf
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014
from galpy.util import conversion, coords

# Allow the caller (e.g. run_all_streams.py) to override the output dir
# so the new DOF-smoothing branch can write to a different location
# without clobbering the original GCV-based results.
_FIGDIR = os.environ.get("GALSTREAM_FIGDIR", "/tmp/galstream_stress_figs")
os.makedirs(_FIGDIR, exist_ok=True)

_mws = None


def get_mws():
    global _mws
    if _mws is None:
        _mws = galstreams.MWStreams(verbose=False, implement_Off=False)
    return _mws


def _cart_to_radec(x, y, z):
    XYZ = coords.galcenrect_to_XYZ(x, y, z, Xsun=8.0, Zsun=0.0208)
    if XYZ.ndim == 2:
        lbd = coords.XYZ_to_lbd(XYZ[:, 0], XYZ[:, 1], XYZ[:, 2], degree=True)
        rad = coords.lb_to_radec(lbd[:, 0], lbd[:, 1], degree=True)
        return rad[:, 0], rad[:, 1]
    lbd = coords.XYZ_to_lbd(XYZ[0], XYZ[1], XYZ[2], degree=True)
    rad = coords.lb_to_radec(lbd[0], lbd[1], degree=True)
    return float(rad[0]), float(rad[1])


def _sky_to_phi12(ra_deg, dec_deg, frame, wrap_center=None):
    ra = np.atleast_1d(np.asarray(ra_deg))
    dec = np.atleast_1d(np.asarray(dec_deg))
    if isinstance(frame, np.ndarray):
        phi12 = coords.radec_to_custom(ra, dec, T=frame, degree=True)
        phi1 = phi12[:, 0]
        phi2 = phi12[:, 1]
    else:
        sc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        sf = sc.transform_to(frame)
        phi1 = sf.phi1.to_value(u.deg)
        phi2 = sf.phi2.to_value(u.deg)
    if wrap_center is not None:
        phi1 = (phi1 - wrap_center + 180.0) % 360.0 - 180.0 + wrap_center
    return phi1, phi2


def _align_rot(prog):
    from galpy.util import _rotate_to_arbitrary_vector

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
    l_pole = np.degrees(np.arctan2(Ly, -Lx))
    b_pole = np.degrees(np.arctan2(Lz, np.sqrt(Lx ** 2 + Ly ** 2)))
    radec = np.asarray(coords.lb_to_radec(l_pole, b_pole, degree=True))
    ra_pole = float(radec[0, 0]) if radec.ndim == 2 else float(radec[0])
    dec_pole = float(radec[0, 1]) if radec.ndim == 2 else float(radec[1])
    L_eq = np.array([
        np.cos(np.radians(dec_pole)) * np.cos(np.radians(ra_pole)),
        np.cos(np.radians(dec_pole)) * np.sin(np.radians(ra_pole)),
        np.sin(np.radians(dec_pole)),
    ])
    return _rotate_to_arbitrary_vector(np.atleast_2d(L_eq), [0.0, 0.0, 1.0])[0]


def _unwrap_around(values, center):
    return ((np.asarray(values) - center + 180.0) % 360.0) - 180.0 + center


def _track_points(track, n=500, to_progenitor=True):
    if to_progenitor:
        lo, hi = track.tp_grid().min(), track.tp_grid().max()
    else:
        lo, hi = _track_data_range(track)
        if hi <= lo:
            lo, hi = track.tp_grid().min(), track.tp_grid().max()
    tps = np.linspace(lo, hi, n)
    x = np.array([track.x(tp) for tp in tps])
    y = np.array([track.y(tp) for tp in tps])
    z = np.array([track.z(tp) for tp in tps])
    R = np.sqrt(x ** 2 + y ** 2)
    ra = np.array([track.ra(tp) for tp in tps])
    dec = np.array([track.dec(tp) for tp in tps])
    return dict(tps=tps, x=x, y=y, z=z, R=R, ra=ra, dec=dec)


def _particles_physical(xv):
    R = xv[0] * 8.0
    phi = xv[5]
    z = xv[3] * 8.0
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    return x, y, z, R


def _track_data_range(track, shrink=0.02):
    lo = float(track.tp_grid().min())
    hi = float(track.tp_grid().max())
    span = hi - lo
    return lo + shrink * span, hi - shrink * span


def _per_tp_width(track, frame, tps, n_samples=250, seed=0, max_abs_phi2=60.0):
    rng = np.random.default_rng(seed)
    phi1_t = np.full(len(tps), np.nan)
    phi2_t = np.full(len(tps), np.nan)
    sig_t = np.full(len(tps), np.nan)
    for i, tp in enumerate(tps):
        ra = track.ra(tp)
        dec = track.dec(tp)
        p1, p2 = _sky_to_phi12(np.array([ra]), np.array([dec]), frame)
        phi1_t[i] = p1[0]
        phi2_t[i] = p2[0]
        if not np.isfinite(p2[0]) or abs(p2[0]) > max_abs_phi2:
            continue
        mean6 = np.array([track.x(tp), track.y(tp), track.z(tp),
                          track.vx(tp), track.vy(tp), track.vz(tp)])
        cov6 = 0.5 * (track.cov(tp) + track.cov(tp).T)
        try:
            samples = rng.multivariate_normal(mean6, cov6, n_samples)
            ra_s, dec_s = _cart_to_radec(samples[:, 0], samples[:, 1],
                                         samples[:, 2])
            _, p2s = _sky_to_phi12(ra_s, dec_s, frame)
            sig_t[i] = np.std(p2s)
        except Exception:
            pass
    return phi1_t, phi2_t, sig_t


def _measure_width_phi2(track, frame, n_samples=250, n_tp=50, seed=0):
    lo, hi = _track_data_range(track)
    if hi <= lo:
        return np.nan, 0
    tps = np.linspace(lo, hi, n_tp)
    _, _, sig_t = _per_tp_width(track, frame, tps, n_samples, seed)
    ok = np.isfinite(sig_t)
    if not ok.any():
        return np.nan, 0
    return float(np.median(sig_t[ok])), int(ok.sum())


def _measure_stream_length(base, sim_frame, prog):
    """Total arc length of the simulated stream in sim-frame phi1
    (approximates what galstreams' track_length measures).

    Computed as the 2–98 percentile phi1 range of the spray particles
    from both arms combined, relative to the progenitor."""
    prog_phi1_raw, _ = _sky_to_phi12(
        np.array([prog.ra()]), np.array([prog.dec()]), sim_frame
    )
    prog_phi1 = float(prog_phi1_raw[0])

    xl, yl, zl, _ = _particles_physical(base["xv_l"])
    xt, yt, zt, _ = _particles_physical(base["xv_t"])
    ra_l, dec_l = _cart_to_radec(xl, yl, zl)
    ra_t, dec_t = _cart_to_radec(xt, yt, zt)

    spans = []
    for ra, dec in ((ra_l, dec_l), (ra_t, dec_t)):
        if len(ra) == 0:
            continue
        p1, _ = _sky_to_phi12(ra, dec, sim_frame)
        p1 = p1 - prog_phi1
        p1 = (p1 + 180.0) % 360.0 - 180.0
        # Arm span: percentile range (excludes outliers/wrapping tails)
        lo, hi = np.percentile(p1, [2, 98])
        spans.append(float(abs(hi - lo)))
    # Combined arc length ~ sum of two arm spans (arms extend on opposite
    # sides of the progenitor)
    return float(sum(spans)) if spans else np.nan


def _sample_tracks(progenitor, mass_msun, tdisrupt_gyr, n, seed_l=42,
                   seed_t=43, smoothing=None, smoothing_factor=1.0):
    M_int = mass_msun / conversion.mass_in_msol(220.0, 8.0)
    spdf_l = fardal15spraydf(pot=MWPotential2014, progenitor=progenitor,
                             tdisrupt=tdisrupt_gyr * u.Gyr, leading=True,
                             progenitor_mass=M_int)
    spdf_t = fardal15spraydf(pot=MWPotential2014, progenitor=progenitor,
                             tdisrupt=tdisrupt_gyr * u.Gyr, leading=False,
                             progenitor_mass=M_int)
    np.random.seed(seed_l)
    xv_l, dt_l = spdf_l._sample_tail(n, integrate=True, leading=True)
    tl = spdf_l.streamTrack(particles=(xv_l, dt_l), tail="leading", order=2,
                            smoothing=smoothing,
                            smoothing_factor=smoothing_factor)
    np.random.seed(seed_t)
    xv_t, dt_t = spdf_t._sample_tail(n, integrate=True, leading=False)
    tt = spdf_t.streamTrack(particles=(xv_t, dt_t), tail="trailing", order=2,
                            smoothing=smoothing,
                            smoothing_factor=smoothing_factor)
    return dict(spdf_l=spdf_l, spdf_t=spdf_t, tl=tl, tt=tt,
                xv_l=xv_l, xv_t=xv_t, dt_l=dt_l, dt_t=dt_t)


def run_stream(track_name, initial_mass_msun=4e4, tdisrupt_gyr=5.0, n=700,
               max_refine_iters=1, conv_tol=0.20, width_scale=0.3,
               match_length=True, show=True, smoothing=None,
               smoothing_factor=1.0):
    """Stress-test streamTrack on a single galstreams stream.

    ``match_length``: also iterate tdisrupt to match galstreams' track
    length. Length scales ≈ linearly with tdisrupt for modest stretches
    (before orbital wrapping saturates). Limits: tdisrupt in [0.3, 12] Gyr.
    """
    t_start = time.time()
    mws = get_mws()
    tr = mws[track_name]
    frame = tr.stream_frame
    target_w_gs = tr.track_width["width_phi2"].to_value("deg")
    target_w = target_w_gs * width_scale
    ref_sc = tr.track
    phi1_ref, phi2_ref = _sky_to_phi12(
        ref_sc.ra.to_value(u.deg), ref_sc.dec.to_value(u.deg), frame
    )
    phi1_ref_u = _unwrap_around(phi1_ref, center=0.0)
    phi1_min, phi1_max = float(phi1_ref_u.min()), float(phi1_ref_u.max())
    target_length = tr.length.to_value("deg")

    print(f"=== {track_name} ===")
    print(f"  target length     : {target_length:.1f} deg")
    print(f"  galstreams phi1   : [{phi1_min:+.1f}, {phi1_max:+.1f}] deg")
    print(f"  galstreams width  : {target_w_gs:.3f} deg")
    print(f"  target (x{width_scale}) : {target_w:.3f} deg")

    prog = Orbit(tr.mid_point)
    sim_frame = _align_rot(prog)

    def _build_and_measure(mass, tdisr):
        b = _sample_tracks(prog, mass, tdisr, n, smoothing=smoothing,
                           smoothing_factor=smoothing_factor)
        sl, nl = _measure_width_phi2(b["tl"], sim_frame)
        st, nt = _measure_width_phi2(b["tt"], sim_frame)
        parts = [(s, k) for s, k in [(sl, nl), (st, nt)]
                 if np.isfinite(s) and k > 0]
        sig = (sum(s * k for s, k in parts) / sum(k for _, k in parts)
               if parts else np.nan)
        L = _measure_stream_length(b, sim_frame, prog)
        return b, sl, st, sig, L

    t0 = time.time()
    base, sig_l0, sig_t0, sig0, len0 = _build_and_measure(
        initial_mass_msun, tdisrupt_gyr
    )
    print(f"  [baseline M={initial_mass_msun:.2e} T={tdisrupt_gyr:.2f}Gyr] "
          f"w={sig0:.3f}° (L={sig_l0:.3f} T={sig_t0:.3f})  len={len0:.1f}°  "
          f"({time.time()-t0:.1f}s)")

    cur = base
    cur_mass = initial_mass_msun
    cur_tdisr = tdisrupt_gyr
    cur_width = sig0
    cur_len = len0
    for i in range(max_refine_iters):
        new_mass = cur_mass
        if np.isfinite(cur_width) and cur_width > 0:
            new_mass = min(max(cur_mass * (target_w / cur_width) ** 3,
                               1e2), 1e10)
        new_tdisr = cur_tdisr
        if match_length and np.isfinite(cur_len) and cur_len > 0:
            new_tdisr = float(np.clip(
                cur_tdisr * (target_length / cur_len), 0.3, 12.0
            ))
        t0 = time.time()
        cur, sig_l, sig_t, sig, L = _build_and_measure(new_mass, new_tdisr)
        cur_mass = new_mass
        cur_tdisr = new_tdisr
        cur_width = sig
        cur_len = L
        print(f"  [refine {i} M={new_mass:.2e} T={new_tdisr:.2f}Gyr] "
              f"w={sig:.3f}° (L={sig_l:.3f} T={sig_t:.3f})  "
              f"len={L:.1f}°  ({time.time()-t0:.1f}s)")
        width_ok = (np.isfinite(sig) and
                    abs(sig - target_w) / target_w < conv_tol)
        length_ok = (not match_length) or (
            np.isfinite(L) and abs(L - target_length) / target_length < 0.30
        )
        if width_ok and length_ok:
            break

    print(f"  final: M={cur_mass:.2e} M_sun  tdisrupt={cur_tdisr:.2f} Gyr  "
          f"total: {time.time()-t_start:.1f}s")

    if show:
        _plot_stream(track_name, tr, frame, target_w, target_w_gs, width_scale,
                     phi1_min, phi1_max, prog, base, cur, initial_mass_msun,
                     cur_mass, sig0, cur_width, target_length,
                     tdisrupt_gyr, cur_tdisr, len0, cur_len)

    return dict(name=track_name, target_width=target_w,
                galstreams_width=target_w_gs, width_scale=width_scale,
                baseline_width=sig0, final_width=cur_width,
                initial_mass=initial_mass_msun, final_mass=cur_mass,
                target_length=target_length,
                baseline_length=len0, final_length=cur_len,
                initial_tdisrupt_gyr=tdisrupt_gyr,
                final_tdisrupt_gyr=cur_tdisr,
                phi1_range=(phi1_min, phi1_max))


def _plot_stream(name, tr, frame, target_w, target_w_gs, width_scale,
                 phi1_min, phi1_max, prog, base, rescaled, m_init, m_final,
                 w_init, w_final, length, t_init=5.0, t_final=5.0,
                 len_init=None, len_final=None):
    phi1_center = 0.5 * (phi1_min + phi1_max)

    ref = tr.track
    ra_ref = ref.ra.to_value(u.deg)
    dec_ref = ref.dec.to_value(u.deg)
    phi1_ref, phi2_ref = _sky_to_phi12(ra_ref, dec_ref, frame)
    phi1_ref_u = _unwrap_around(phi1_ref, center=phi1_center)

    sim_frame = _align_rot(prog)
    sim_phi1_center = 0.0

    tl_r = _track_points(rescaled["tl"])
    tt_r = _track_points(rescaled["tt"])
    phi1_tl, phi2_tl = _sky_to_phi12(tl_r["ra"], tl_r["dec"], frame)
    phi1_tt, phi2_tt = _sky_to_phi12(tt_r["ra"], tt_r["dec"], frame)
    phi1_tl_u = _unwrap_around(phi1_tl, center=phi1_center)
    phi1_tt_u = _unwrap_around(phi1_tt, center=phi1_center)

    xl, yl, zl, Rl = _particles_physical(rescaled["xv_l"])
    xt, yt, zt, Rt = _particles_physical(rescaled["xv_t"])
    ra_l, dec_l = _cart_to_radec(xl, yl, zl)
    ra_t, dec_t = _cart_to_radec(xt, yt, zt)

    prog_phi1_raw, prog_phi2 = _sky_to_phi12(
        np.array([prog.ra()]), np.array([prog.dec()]), sim_frame
    )
    prog_phi1 = float(prog_phi1_raw[0])

    def to_sim(ra, dec):
        p1, p2 = _sky_to_phi12(ra, dec, sim_frame, wrap_center=prog_phi1)
        return p1 - prog_phi1, p2

    phi1_sim_l, phi2_sim_l = to_sim(ra_l, dec_l)
    phi1_sim_t, phi2_sim_t = to_sim(ra_t, dec_t)

    def arm_band(track):
        # Use the full tp_grid range so the plotted track reaches the
        # progenitor at tp=0. Sequentially unwrap phi1 along tp so
        # wrapping streams draw as a continuous curve rather than
        # jumping ±360° between consecutive samples.
        lo = float(track.tp_grid().min())
        hi = float(track.tp_grid().max())
        if hi <= lo:
            return (np.array([]),) * 3
        tps_dense = np.linspace(lo, hi, 120)
        p1_raw, p2_raw, s = _per_tp_width(track, sim_frame, tps_dense, 150)
        p1 = p1_raw - prog_phi1
        # Start the sequence near phi1=0 then unwrap
        p1 = (p1 + 180.0) % 360.0 - 180.0
        p1 = np.rad2deg(np.unwrap(np.deg2rad(p1)))
        # For leading (tp >= 0) the first tp is the progenitor end; for
        # trailing (tp <= 0) the last tp is. Shift the full sequence so
        # the progenitor-end lands near phi1=0 (it may have drifted by
        # ±360k during unwrap). Derive arm sign from the public tp grid
        # (the StreamTrack base class no longer carries _arm_sign post the
        # base/fitter split refactor).
        tp_grid = track.tp_grid()
        if tp_grid[0] >= 0.0:  # leading: tp in [0, tp_hi]
            ref_idx = 0
        else:  # trailing: tp in [tp_lo, 0]
            ref_idx = -1
        shift = 360.0 * np.round(p1[ref_idx] / 360.0)
        p1 = p1 - shift
        return p1, p2_raw, s

    p1bl_u, p2bl, sbl = arm_band(rescaled["tl"])
    p1bt_u, p2bt, sbt = arm_band(rescaled["tt"])

    ra_gs = tr.track.ra.to_value(u.deg)
    dec_gs = tr.track.dec.to_value(u.deg)
    gs_sim_phi1, gs_sim_phi2 = to_sim(ra_gs, dec_gs)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0, 0]
    ax.fill_between(phi1_ref_u, phi2_ref - target_w, phi2_ref + target_w,
                    color="k", alpha=0.15,
                    label=f"scaled target ±{target_w:.2f}°")
    ax.fill_between(phi1_ref_u, phi2_ref - target_w_gs,
                    phi2_ref + target_w_gs, color="k", alpha=0.05,
                    label=f"galstreams ±{target_w_gs:.2f}°")
    ax.plot(phi1_ref_u, phi2_ref, "k-", lw=2.5,
            label=f"galstreams {name}")
    ax.set_xlabel(r"$\phi_1$ [deg]")
    ax.set_ylabel(r"$\phi_2$ [deg]")
    ax.set_title(f"galstreams track  (width = {target_w_gs:.2f}°  →  "
                 f"target = {target_w:.2f}° @ x{width_scale})")
    y_lo = float(np.nanmin(phi2_ref - target_w_gs))
    y_hi = float(np.nanmax(phi2_ref + target_w_gs))
    y_pad = 0.1 * (y_hi - y_lo) + 0.2
    ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3)

    # --- (phi1, phi2) simulated streamTrack in progenitor-L-aligned frame ---
    ax = axes[0, 1]
    ax.scatter(phi1_sim_l, phi2_sim_l, s=2, c="tab:red", alpha=0.3,
               zorder=0, label="leading particles")
    ax.scatter(phi1_sim_t, phi2_sim_t, s=2, c="tab:blue", alpha=0.3,
               zorder=0, label="trailing particles")

    def plot_track_band(phi1, phi2, sig, colour, label):
        finite = np.isfinite(phi1) & np.isfinite(phi2) & np.isfinite(sig)
        ph1 = phi1[finite]; ph2 = phi2[finite]; s = sig[finite]
        if len(ph1) < 2:
            return
        poly_x = np.concatenate([ph1, ph1[::-1]])
        poly_y = np.concatenate([ph2 + s, (ph2 - s)[::-1]])
        ax.fill(poly_x, poly_y, color=colour, alpha=0.35, zorder=3,
                linewidth=0)
        ax.plot(ph1, ph2, color=colour, lw=1.8, zorder=4, label=label)

    plot_track_band(p1bl_u, p2bl, sbl, "darkred", "leading track ±σ")
    plot_track_band(p1bt_u, p2bt, sbt, "navy", "trailing track ±σ")

    all_p2 = []
    for phi2, sig in [(p2bl, sbl), (p2bt, sbt)]:
        if len(phi2):
            ok = np.isfinite(phi2) & np.isfinite(sig)
            if ok.any():
                all_p2.append(phi2[ok] - sig[ok])
                all_p2.append(phi2[ok] + sig[ok])
    for arr in (phi2_sim_l, phi2_sim_t):
        if len(arr):
            ok = np.isfinite(arr)
            if ok.any():
                med = np.median(arr[ok])
                mad = np.median(np.abs(arr[ok] - med)) or 0.5
                all_p2.append(np.array([med - 5 * 1.48 * mad,
                                        med + 5 * 1.48 * mad]))
    if all_p2:
        y_vals = np.concatenate(all_p2)
        y_lo, y_hi = float(np.nanmin(y_vals)), float(np.nanmax(y_vals))
        y_pad = 0.10 * (y_hi - y_lo) + 0.2
        ax.set_ylim(y_lo - y_pad, y_hi + y_pad)

    all_p1 = []
    for arr in (phi1_sim_l, phi1_sim_t, p1bl_u, p1bt_u):
        if len(arr):
            ok = np.isfinite(arr)
            if ok.any():
                all_p1.append(arr[ok])
    if all_p1:
        xvals = np.concatenate(all_p1)
        x_lo, x_hi = float(xvals.min()), float(xvals.max())
        x_pad = 0.05 * (x_hi - x_lo) + 0.5
        ax.set_xlim(x_lo - x_pad, x_hi + x_pad)

    ax.plot(gs_sim_phi1, gs_sim_phi2, "k-", lw=1.5, alpha=0.8, zorder=2,
            label="galstreams ref.")

    ax.scatter([0.0], [float(prog_phi2[0])], s=80, c="k", marker="*",
               zorder=5, label="progenitor")
    ax.set_xlabel(r"$\phi_1 - \phi_{1,{\rm prog}}$ [deg]   (progenitor-L aligned)")
    ax.set_ylabel(r"$\phi_2$ [deg]")
    ax.set_title(f"streamTrack in progenitor-L frame\n"
                 f"(width = {w_final:.2f}°,  M = {m_final:.2e} M☉)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3)

    def auto_limits(ax_, *arrays, pad_frac=0.05):
        xs, ys = [], []
        for xv, yv in arrays:
            xv = np.asarray(xv); yv = np.asarray(yv)
            ok = np.isfinite(xv) & np.isfinite(yv)
            if ok.any():
                xs.append(xv[ok]); ys.append(yv[ok])
        if not xs:
            return
        xs = np.concatenate(xs); ys = np.concatenate(ys)
        xr = xs.max() - xs.min(); yr = ys.max() - ys.min()
        ax_.set_xlim(xs.min() - pad_frac * xr - 1e-6,
                     xs.max() + pad_frac * xr + 1e-6)
        ax_.set_ylim(ys.min() - pad_frac * yr - 1e-6,
                     ys.max() + pad_frac * yr + 1e-6)

    ax = axes[0, 2]
    ax.scatter(ra_l, dec_l, s=2, c="tab:red", alpha=0.3)
    ax.scatter(ra_t, dec_t, s=2, c="tab:blue", alpha=0.3)
    ax.plot(ra_ref, dec_ref, "k-", lw=2.5, label="galstreams")
    ax.plot(tl_r["ra"], tl_r["dec"], color="tab:red", lw=1.8, label="leading")
    ax.plot(tt_r["ra"], tt_r["dec"], color="tab:blue", lw=1.8, label="trailing")
    ax.scatter([prog.ra()], [prog.dec()], s=80, c="k", marker="*",
               zorder=5, label="progenitor")
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.set_title("equatorial sky (full 5 Gyr simulation)")
    auto_limits(ax, (ra_l, dec_l), (ra_t, dec_t), (ra_ref, dec_ref),
                (tl_r["ra"], tl_r["dec"]), (tt_r["ra"], tt_r["dec"]),
                ([prog.ra()], [prog.dec()]))
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.scatter(xl, yl, s=2, c="tab:red", alpha=0.3)
    ax.scatter(xt, yt, s=2, c="tab:blue", alpha=0.3)
    ax.plot(tl_r["x"], tl_r["y"], color="tab:red", lw=1.8, label="leading")
    ax.plot(tt_r["x"], tt_r["y"], color="tab:blue", lw=1.8, label="trailing")
    ax.scatter([prog.x()], [prog.y()], s=80, c="k", marker="*", zorder=5,
               label="progenitor")
    ax.set_xlabel("x [kpc]   (galpy: Sun at +x)")
    ax.set_ylabel("y [kpc]")
    ax.set_title("Galactocentric (x, y)")
    auto_limits(ax, (xl, yl), (xt, yt), (tl_r["x"], tl_r["y"]),
                (tt_r["x"], tt_r["y"]), ([prog.x()], [prog.y()]))
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.scatter(Rl, zl, s=2, c="tab:red", alpha=0.3)
    ax.scatter(Rt, zt, s=2, c="tab:blue", alpha=0.3)
    ax.plot(tl_r["R"], tl_r["z"], color="tab:red", lw=1.8, label="leading")
    ax.plot(tt_r["R"], tt_r["z"], color="tab:blue", lw=1.8, label="trailing")
    ax.scatter([prog.R()], [prog.z()], s=80, c="k", marker="*", zorder=5,
               label="progenitor")
    ax.set_xlabel("R [kpc]")
    ax.set_ylabel("z [kpc]")
    ax.set_title("Galactocentric (R, z)")
    auto_limits(ax, (Rl, zl), (Rt, zt), (tl_r["R"], tl_r["z"]),
                (tt_r["R"], tt_r["z"]), ([prog.R()], [prog.z()]))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    labels = [f"galstreams\n(raw)", f"target\n(×{width_scale})",
              f"baseline\n{m_init:.1e} M☉", f"rescaled\n{m_final:.1e} M☉"]
    values = [target_w_gs, target_w, w_init, w_final]
    colors = ["lightgray", "k", "tab:gray", "tab:green"]
    ax.bar(labels, values, color=colors, alpha=0.75)
    ax.axhline(target_w, color="k", ls="--", alpha=0.5, label="target")
    ax.axhline(target_w_gs, color="lightgray", ls=":", alpha=0.7)
    ax.set_ylabel(r"width $\sigma_{\phi_2}$ [deg]")
    ax.set_title("width comparison")
    ax.grid(axis="y", alpha=0.3)
    vmax = max(v for v in values if np.isfinite(v))
    ax.set_ylim(0, 1.15 * vmax)
    for i, v in enumerate(values):
        if np.isfinite(v):
            ax.text(i, v + 0.02 * vmax, f"{v:.2f}°", ha="center", fontsize=9)

    len_str = (f"len: {len_init:.0f}° → {len_final:.0f}° (target {length:.0f}°)"
               if len_init is not None and len_final is not None
               else f"L_target={length:.0f}°")
    fig.suptitle(
        f"{name}:  M: {m_init:.1e} → {m_final:.1e} M☉  "
        f"(w: {w_init:.2f}° → {w_final:.2f}°, target {target_w:.2f}°)  |  "
        f"tdisrupt: {t_init:.1f} → {t_final:.1f} Gyr  "
        f"({len_str})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(f"{_FIGDIR}/{name}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def list_decent_streams():
    mws = get_mws()
    out = []
    for name in mws.summary.index:
        try:
            tr = mws[name]
        except KeyError:
            continue
        flags = tr.InfoFlags
        w = tr.track_width.get("width_phi2")
        if w is None:
            continue
        wv = w.to_value("deg")
        if wv < 0.05:
            continue
        if flags[2] != "1":
            continue
        if flags[1] not in ("1", "2"):
            continue
        if tr.length.to_value("deg") < 3.0:
            continue
        out.append((name, tr.length.to_value("deg")))
    out.sort(key=lambda x: -x[1])
    return [n for n, _ in out]
