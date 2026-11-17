"""
Linear reference for the DPS comparison: tomographic Wiener filters on the same kind of mocks.

For Gaussian fields the posterior mean is the Wiener filter, so it predicts what a sampler with
the right two-point prior does to the map amplitude when the IA term is in or out of its forward
model. Per Fourier mode, with the n_bins x n_bins spectra at A_IA = 1, noise N, and
M = 1 + A R (the sampler's forward model, kappa -> kappa + A R kappa):
    ignored    F = C_kk [C_kk + N]^-1                       model: data = kappa + noise
    modelled   F = C_kk M^T [M C_kk M^T + N]^-1             model: data = M kappa + noise
Both are applied to data = kappa + A kappa_IA(shells) + noise for every held-out map, and the
same statistics as analyze_dps.py are computed: the slope of the estimate against the truth, the
relative change of the slope when IA is ignored, and the leak of the injected IA into the map.
Also: the same with only one tomographic bin at a time (no information from the other bins),
which is where the single-bin Kaiser-Squires expectation applies.

Usage (from the top of this repository):
    python3 des_y3_like/wiener_reference.py
    python3 des_y3_like/wiener_reference.py --neff 7.5 --out des_y3_like/results/wiener_reference_deep.json
"""
import argparse
import json
import os

import numpy as np
import yaml

p = argparse.ArgumentParser()
p.add_argument('--config', default='des_y3_like/des_like.yaml')
p.add_argument('--work', default='work')
p.add_argument('--out', default='des_y3_like/results/wiener_reference.json')
p.add_argument('--seed', type=int, default=11)
p.add_argument('--neff', type=float, default=None, help='galaxies per arcmin^2 per bin (default: the config value)')
a = p.parse_args()

cfg = yaml.safe_load(open(a.config))
info = json.load(open(os.path.join(a.work, 'maps_info.json')))
n, theta, nb = cfg['map']['n_grid'], cfg['map']['theta_max'], cfg['map']['n_tomo']
A = float(cfg['ia']['A_IA'])
neff = a.neff if a.neff is not None else cfg['data']['neff']
sigma = cfg['data']['sigma_e'] / np.sqrt(neff * (theta * 60 / n) ** 2)
sp = np.load(cfg['ia']['spectra_file'])
dx = np.radians(theta) / n
l1 = 2 * np.pi * np.fft.fftfreq(n, d=dx)
ell = np.hypot(l1[:, None], l1[None, :])
le = np.log(np.clip(ell, sp['ell'][0], sp['ell'][-1]))
grid = lambda C: np.stack([np.stack([np.interp(le, np.log(sp['ell']), C[:, i, j]) for j in range(nb)], -1)
                           for i in range(nb)], -2)                 # (n, n, nb, nb)
Ckk, CIk = grid(sp['C_kk']), grid(sp['C_Ik'])
N = sigma ** 2 * dx ** 2 * np.eye(nb)                               # noise power of the KS map of one bin
R = CIk @ np.linalg.inv(Ckk)
M = np.eye(nb) + A * R
Mt = np.swapaxes(M, -1, -2)
F = dict(ignored=Ckk @ np.linalg.inv(Ckk + N),
         modelled=Ckk @ Mt @ np.linalg.inv(M @ Ckk @ Mt + N))
# one bin at a time: diagonal spectra only
d = lambda X: np.einsum('xybb->xyb', X)
Md = 1 + A * d(R * 0 + CIk) / d(Ckk)                                # single-bin R_bb = C_Ik,bb / C_kk,bb
F1 = dict(ignored=d(Ckk) / (d(Ckk) + N[0, 0]), modelled=d(Ckk) * Md / (Md ** 2 * d(Ckk) + N[0, 0]))
for f in list(F.values()) + list(F1.values()):
    f[0, 0] = 0

apply = lambda Fm, x: np.fft.ifft2(np.einsum('xybc,cxy->bxy', Fm, np.fft.fft2(x))).real
apply1 = lambda Fm, x: np.fft.ifft2(np.moveaxis(Fm, -1, 0) * np.fft.fft2(x)).real
zm = lambda x: x - x.mean(axis=(-2, -1), keepdims=True)
slope = lambda m, t: (zm(m) * zm(t)).sum(axis=(-2, -1)) / (zm(t) ** 2).sum(axis=(-2, -1))
corr = lambda m, t: (zm(m) * zm(t)).sum(axis=(-2, -1)) / np.sqrt((zm(m) ** 2).sum(axis=(-2, -1)) * (zm(t) ** 2).sum(axis=(-2, -1)))

rng = np.random.default_rng(a.seed)
rows = {k: [] for k in ('slope_modelled', 'slope_ignored', 'leak', 'corr_modelled', 'corr_ignored',
                        'slope1_modelled', 'slope1_ignored')}
for j in range(info['n_test']):
    t = np.load(os.path.join(a.work, 'test', f'kappa_{j}.npy')).astype(float)
    kia = np.load(os.path.join(a.work, 'test', f'ia_{j}.npy')).astype(float)
    data = t + A * kia + sigma * rng.standard_normal(t.shape)      # KS map of the noisy shear: white noise sigma
    est = {k: apply(F[k], data) for k in F}
    est1 = {k: apply1(F1[k], data) for k in F1}
    rows['slope_modelled'].append(slope(est['modelled'], t))
    rows['slope_ignored'].append(slope(est['ignored'], t))
    rows['corr_modelled'].append(corr(est['modelled'], t))
    rows['corr_ignored'].append(corr(est['ignored'], t))
    rows['leak'].append(slope(est['ignored'] - est['modelled'], A * kia))
    rows['slope1_modelled'].append(slope(est1['modelled'], t))
    rows['slope1_ignored'].append(slope(est1['ignored'], t))
r = {k: np.array(v) for k, v in rows.items()}
ms = lambda x: (x.mean(0).tolist(), (x.std(0, ddof=1) / np.sqrt(len(x))).tolist())
out = dict(
    about='wiener_reference.py: tomographic and single-bin Wiener filters on the held-out maps with shell-built IA, '
          'A = %g, %g galaxies per arcmin^2 per bin. (mean, standard error) over %d maps, per tomographic bin.' % (A, neff, info['n_test']),
    neff=neff, sigma_pixel=float(sigma),
    slope_modelled=ms(r['slope_modelled']), slope_ignored=ms(r['slope_ignored']),
    slope_ratio_minus_1=ms(r['slope_ignored'] / r['slope_modelled'] - 1),
    corr_modelled=ms(r['corr_modelled']), corr_ignored=ms(r['corr_ignored']), leak=ms(r['leak']),
    single_bin_slope_ratio_minus_1=ms(r['slope1_ignored'] / r['slope1_modelled'] - 1),
    single_bin_slope_modelled=ms(r['slope1_modelled']))
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(out, open(a.out, 'w'), indent=1)
np.set_printoptions(precision=3, suppress=True, linewidth=150)
for k in ('slope_modelled', 'slope_ignored', 'slope_ratio_minus_1', 'corr_modelled', 'corr_ignored', 'leak',
          'single_bin_slope_modelled', 'single_bin_slope_ratio_minus_1'):
    print(f'{k:32s} mean {np.array(out[k][0])}   sem {np.array(out[k][1])}')
