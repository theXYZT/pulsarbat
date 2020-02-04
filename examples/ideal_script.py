import pulsarbat as pb
import baseband
import astropy.units as u
from pathlib import Path

# Getting baseband file handle
folder = Path('/mnt/scratch-lustre/mahajan/Data/Ar_P3229/B1937+21_58245/gpu09')
fs = sorted(folder.glob('*.raw'))
fh = baseband.open(fs, 'rs', format='guppi')

obs = pb.PUPPIObservation(fh)
DM = pb.DispersionMeasure(71.02227638)
ref_freq = 375.4375 * u.MHz
polyco = pb.Polyco("/mnt/scratch-lustre/mahajan/Timing/B1937+21_58245.dat")

z = obs.read(2**24)
z = pb.transforms.dedisperse(z, DM, ref_freq)
y = pb.reductions.to_intensity(z)

pp = pb.pulsar.fold(y, polyco, ngate=256)
