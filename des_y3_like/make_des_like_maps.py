"""
DES Y3-like convergence maps with field-level NLA intrinsic alignments, from lognormal shells.

Shell statistics: delta_i = lam_i * (exp(g - var_g / 2) - 1) with g Gaussian (a shifted
lognormal). lam_i = 1 is the ordinary lognormal (delta >= -1). At 8 arcmin pixels the nearest
shells have a pixel variance of several, where an ordinary lognormal puts single pixels at
delta of order 100 and kappa of order 0.5; that breaks the min-max normalization of the
diffusion prior and has nothing to do with IA. So lam_i = max(1, rms_i / --max_log_rms): the
variance of exp(g) is capped, the power spectrum of every shell is unchanged, and the maps
stay mildly skewed. Shells with lam_i > 1 are a statistical model, not a physical density.

The lightcone is cut into thin shells in comoving distance. Shell i gets a lognormal density
contrast delta_i(theta) whose angular power spectrum is the Limber spectrum of a thin shell,
    C_i(l) = P_nl(k = (l + 1/2) / chi_i, z_i) / (chi_i^2 dchi_i),
and the maps of tomographic bin b are weighted sums over the SAME shells:
    kappa_b    = sum_i W_L[b, i] delta_i      lensing weights   (matter in front of the sources)
    kappa_IA_b = sum_i W_I[b, i] delta_i      NLA weights, A_IA = 1 (matter at the sources)
This is the construction of training_data_creation/create_desy3_training_patches_ia.py with
simulated particle shells replaced by lognormal shells. The IA map is built from the density
field itself, so it is independent of the R(l) kappa + eps model used by the sampler.

Outputs (in --out):
    patches/{1..n_train}.npy      (n_bins, npix, npix) float32 kappa, the training set
    test/kappa_{j}.npy, test/ia_{j}.npy   held-out maps with their IA maps (A_IA = 1)
    maps_info.json                grid, cosmology, kappa_min / kappa_max of the training set

Usage:
    python3 make_des_like_maps.py --dps_code <.../denoising_diffusion_pytorch> --nz <n(z) file> --out work
"""
import argparse
import json
import os
import sys

import numpy as np
import pyccl as ccl

p = argparse.ArgumentParser()
p.add_argument('--dps_code', required=True, help='path to DPS_mass_mapping/denoising_diffusion_pytorch')
p.add_argument('--nz', required=True, help='source n(z): DES Y3 2pt FITS or Cocoa text file')
p.add_argument('--out', default='work')
p.add_argument('--n_train', type=int, default=4000)
p.add_argument('--n_test', type=int, default=100)
p.add_argument('--npix', type=int, default=64)
p.add_argument('--theta_max', type=float, default=8.53333333333, help='patch side in degrees')
p.add_argument('--n_shells', type=int, default=100)
p.add_argument('--max_log_rms', type=float, default=0.5,
               help='cap on rms(delta) / lam of each shell (0.5: variance of g <= ln 1.25)')
p.add_argument('--z_min', type=float, default=0.0,
               help='drop shells nearer than this redshift (the repo drops its 15 nearest simulation shells)')
p.add_argument('--seed', type=int, default=2026)
p.add_argument('--Omega_c', type=float, default=0.25)
p.add_argument('--Omega_b', type=float, default=0.05)
p.add_argument('--h', type=float, default=0.68)
p.add_argument('--sigma8', type=float, default=0.81)
p.add_argument('--n_s', type=float, default=0.965)
a = p.parse_args()

sys.path.insert(0, a.dps_code)
from nla_shell_weights import nla_shell_weights, lensing_shell_weights   # noqa: E402
from nz_io import load_nz                                                # noqa: E402

cosmo = ccl.Cosmology(Omega_c=a.Omega_c, Omega_b=a.Omega_b, h=a.h, sigma8=a.sigma8, n_s=a.n_s,
                      matter_power_spectrum='halofit', transfer_function='eisenstein_hu')
z_nz, nz_bins = load_nz(a.nz)
nb = len(nz_bins)

# shells, uniform in comoving distance out to the end of n(z)
chi_e = np.linspace(1.0, ccl.comoving_radial_distance(cosmo, 1 / (1 + z_nz[-1])), a.n_shells + 1)   # Mpc
z_e = 1 / ccl.scale_factor_of_chi(cosmo, chi_e) - 1
chi, dchi = 0.5 * (chi_e[1:] + chi_e[:-1]), np.diff(chi_e)
a_mid = ccl.scale_factor_of_chi(cosmo, chi)
W_L = lensing_shell_weights(z_e[:-1], z_e[1:], z_nz, nz_bins, cosmo)             # (nb, n_shells)
W_I = nla_shell_weights(z_e[:-1], z_e[1:], z_nz, nz_bins, cosmo, A_IA=1.0)       # (nb, n_shells)
keep = z_e[:-1] >= a.z_min                                                       # nearest shells dropped
W_L[:, ~keep] = 0.0
W_I[:, ~keep] = 0.0

# flat-sky grid
n = a.npix
dx = np.radians(a.theta_max) / n
l1 = 2 * np.pi * np.fft.fftfreq(n, d=dx)
ell = np.hypot(l1[:, None], l1[None, :])
A_pix = dx ** 2

# per shell: target power of delta, and the power of the Gaussian field g with
# delta = exp(g - var_g / 2) - 1  (xi_g = ln(1 + xi_delta))
sqrtPg = np.zeros((a.n_shells, n, n))
lam = np.ones(a.n_shells)
var_g = np.zeros(a.n_shells)
var_delta = np.zeros(a.n_shells)
clipped = 0.0
for i in np.flatnonzero(keep):
    C = ccl.nonlin_matter_power(cosmo, (ell.ravel() + 0.5) / chi[i], a_mid[i]).reshape(n, n) / (chi[i] ** 2 * dchi[i])
    C[0, 0] = 0.0
    xi = np.fft.ifft2(C / A_pix).real                    # correlation function of delta on the grid
    var_delta[i] = xi[0, 0]
    lam[i] = max(1.0, np.sqrt(xi[0, 0]) / a.max_log_rms)
    Pg = np.fft.fft2(np.log(np.maximum(1.0 + xi / lam[i] ** 2, 1e-3))).real    # variance of g per ortho-FFT mode
    Pg[0, 0] = 0.0
    clipped = max(clipped, -Pg.min() / Pg.max())
    Pg = np.clip(Pg, 0.0, None)
    sqrtPg[i] = np.sqrt(Pg)
    var_g[i] = Pg.sum() / n ** 2
print(f'{a.n_shells} shells to z = {z_e[-1]:.2f}, {nb} source bins, grid {n}^2 x {a.theta_max * 60 / n:.1f} arcmin')
print(f'{keep.sum()} shells kept (z >= {a.z_min}); '
      f'pixel variance of delta per kept shell: {var_delta[keep].min():.3f} .. {var_delta[keep].max():.3f};  '
      f'largest negative Gaussian power clipped: {clipped:.1e} of the peak')

rng = np.random.default_rng(a.seed)


def draw():
    """One lightcone: kappa and kappa_IA (A_IA = 1), each (nb, n, n)."""
    w = np.fft.fft2(rng.standard_normal((a.n_shells, n, n)), norm='ortho')
    g = np.fft.ifft2(w * sqrtPg, norm='ortho').real
    delta = lam[:, None, None] * (np.exp(g - 0.5 * var_g[:, None, None]) - 1.0)
    return np.einsum('bi,ixy->bxy', W_L, delta), np.einsum('bi,ixy->bxy', W_I, delta)


os.makedirs(os.path.join(a.out, 'patches'), exist_ok=True)
os.makedirs(os.path.join(a.out, 'test'), exist_ok=True)
kmin, kmax = np.full(nb, np.inf), np.full(nb, -np.inf)
rms = np.zeros(nb)
for j in range(1, a.n_train + 1):
    kappa, _ = draw()
    np.save(os.path.join(a.out, 'patches', f'{j}.npy'), kappa.astype(np.float32))
    kmin, kmax = np.minimum(kmin, kappa.min(axis=(1, 2))), np.maximum(kmax, kappa.max(axis=(1, 2)))
    rms += kappa.var(axis=(1, 2)) / a.n_train
for j in range(a.n_test):
    kappa, ia = draw()
    np.save(os.path.join(a.out, 'test', f'kappa_{j}.npy'), kappa.astype(np.float32))
    np.save(os.path.join(a.out, 'test', f'ia_{j}.npy'), ia.astype(np.float32))

# Limber spectra of exactly these shells (same format as make_ia_spectra.py), for the sampler's R(l)
ell_tab = np.geomspace(10, 3e4, 300)
D = np.array([ccl.nonlin_matter_power(cosmo, (ell_tab + 0.5) / c, am) / (c ** 2 * dc) if k else np.zeros_like(ell_tab)
              for c, dc, am, k in zip(chi, dchi, a_mid, keep)]).T                      # (L, n_shells)
gram = lambda X, Y: np.einsum('ai,bi,li->lab', X, Y, D)
np.savez(os.path.join(a.out, 'ia_spectra_shells.npz'), ell=ell_tab, C_kk=gram(W_L, W_L), C_Ik=gram(W_I, W_L), C_II=gram(W_I, W_I))

info = dict(n_bins=nb, npix=n, theta_max_deg=a.theta_max, pixel_arcmin=a.theta_max * 60 / n,
            n_shells=a.n_shells, n_shells_kept=int(keep.sum()), z_min=a.z_min, z_max=float(z_e[-1]), n_train=a.n_train, n_test=a.n_test, seed=a.seed,
            cosmology=dict(Omega_c=a.Omega_c, Omega_b=a.Omega_b, h=a.h, sigma8=a.sigma8, n_s=a.n_s),
            nz_file=os.path.basename(a.nz),
            kappa_min=kmin.tolist(), kappa_max=kmax.tolist(), kappa_rms=np.sqrt(rms).tolist(),
            max_log_rms=a.max_log_rms, n_shells_shifted=int((lam > 1).sum()),
            shell_var_delta_max=float(var_delta.max()), lognormal_clipped_fraction=float(clipped))
json.dump(info, open(os.path.join(a.out, 'maps_info.json'), 'w'), indent=1)
fmt = lambda v: ','.join(f'{x:.6f}' for x in v)
print(f'wrote {a.n_train} training and {a.n_test} test maps to {a.out}')
print('kappa rms per bin:', np.round(np.sqrt(rms), 5))
print(f'kappa_min: {fmt(kmin)}')
print(f'kappa_max: {fmt(kmax)}')
