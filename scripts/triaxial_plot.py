import matplotlib; matplotlib.use('Agg')
import numpy, sys
from matplotlib import pyplot
from astropy import units as u
from galpy.df import fardal15spraydf
from galpy.orbit import Orbit
from galpy.potential import TriaxialNFWPotential
from galpy.util import _rotate_to_arbitrary_vector, coords

version = sys.argv[1]  # "v1" or "v2"
tp = TriaxialNFWPotential(normalize=1., a=20./8., b=0.8, c=0.6)
prog = Orbit([1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596], ro=8., vo=220.)
sun = Orbit([0,0,0,0,0,0], radec=True); sun.vxvv[0][1]=sun.vxvv[0][2]=sun.vxvv[0][4]=0.; sun.turn_physical_off()
p = prog(); p.turn_physical_off()
dx,dy,dz = p.x()-sun.x(), p.y()-sun.y(), p.z()-sun.z()
dvx,dvy,dvz = p.vx()-sun.vx(), p.vy()-sun.vy(), p.vz()-sun.vz()
Lx=dy*dvz-dz*dvy; Ly=dz*dvx-dx*dvz; Lz=dx*dvy-dy*dvx
radec_pole = coords.lb_to_radec(
    numpy.degrees(numpy.arctan2(Ly, -Lx)),
    numpy.degrees(numpy.arctan2(Lz, numpy.sqrt(Lx**2+Ly**2))), degree=True)
L_eq = numpy.array([numpy.cos(numpy.radians(radec_pole[1]))*numpy.cos(numpy.radians(radec_pole[0])),
                    numpy.cos(numpy.radians(radec_pole[1]))*numpy.sin(numpy.radians(radec_pole[0])),
                    numpy.sin(numpy.radians(radec_pole[1]))])
rot = _rotate_to_arbitrary_vector(numpy.atleast_2d(L_eq), [0.,0.,1.])[0]
def to_phi12(ra, dec):
    ra = numpy.asarray(getattr(ra, 'value', ra))
    dec = numpy.asarray(getattr(dec, 'value', dec))
    phi12 = coords.radec_to_custom(ra, dec, T=rot, degree=True)
    phi1, phi2 = phi12[:,0], phi12[:,1]
    med = numpy.median(phi1)
    return (phi1-med+180.)%360.-180.+med, phi2

for mass, label in [(1e4, "cold_1e4"), (1e7, "warm_1e7")]:
    print(f"{version} {label}...")
    spdf = fardal15spraydf(progenitor_mass=mass*u.Msun, progenitor=prog, pot=tp,
                          tdisrupt=5.0*u.Gyr, tail='both')
    numpy.random.seed(7)
    track = spdf.streamTrack(n=1000)
    samples = spdf.sample(n=1000)
    fig, axes = pyplot.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{version} stream-orbit (TriaxialNFW, 5 Gyr) — {label}', fontsize=14)
    ax = axes[0]; pyplot.sca(ax)
    phi1_s, phi2_s = to_phi12(samples.ra(), samples.dec())
    ax.scatter(phi1_s, phi2_s, s=2, alpha=0.3, color='0.6')
    for arm, col in [(track.leading, 'C3'), (track.trailing, 'C0')]:
        p1, p2 = to_phi12(arm.ra(arm.tp_grid()), arm.dec(arm.tp_grid()))
        ax.plot(p1, p2, col, lw=1.5)
    med2 = numpy.median(phi2_s); span2 = max(numpy.percentile(phi2_s,99)-numpy.percentile(phi2_s,1), 4.)
    ax.set_ylim(med2-1.5*span2, med2+1.5*span2)
    ax.set_xlabel('phi1'); ax.set_ylabel('phi2'); ax.set_title('Aligned')
    ax = axes[1]; pyplot.sca(ax)
    ax.scatter(samples.x(), samples.y(), s=2, alpha=0.3, color='0.6')
    track.leading.plot(d1='x', d2='y', color='C3')
    track.trailing.plot(d1='x', d2='y', color='C0')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_title('(x, y)')
    ax = axes[2]; pyplot.sca(ax)
    ax.scatter(samples.R(), samples.z(), s=2, alpha=0.3, color='0.6')
    track.leading.plot(d1='R', d2='z', color='C3')
    track.trailing.plot(d1='R', d2='z', color='C0')
    ax.set_xlabel('R'); ax.set_ylabel('z'); ax.set_title('(R, z)')
    pyplot.tight_layout()
    out = f'doc_images/triaxial_{version}_{label}.png'
    pyplot.savefig(out, dpi=120); pyplot.close(fig)
    print(f'  wrote {out}')
