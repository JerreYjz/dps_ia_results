"""
Test the R(l) kappa + eps IA model against IA maps built independently from the density shells.

No diffusion model is involved. For every held-out map of make_des_like_maps.py:
  1. spectra: measured C_kk, C_Ik, C_II against the Limber spectra the sampler uses;
  2. prediction: kappa_IA_pred = R(l) kappa against the shell-built kappa_IA: fraction of the IA
     variance that is predicted, against the theory value, and the correlation of the residual
     with kappa (should vanish);
  3. residual: power of (kappa_IA - R kappa) against the model's S(l);
  4. amplitude: noisy DES Y3-like shear is made from kappa + A * kappa_IA(shells) and the exact
     Gaussian conditional p(A | true kappa, data) is evaluated. Its mean should scatter around
     the injected A with the reported width. The same is done without noise (the effect of the
     stochastic part eps alone) and with 10 noise draws per map (the pull width).

Usage (from the top of this repository, in the Cocoa environment):
    DPS_DEVICE=cpu python3 des_y3_like/validate_ia_model.py --dps_code <.../denoising_diffusion_pytorch>
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

p = argparse.ArgumentParser()
p.add_argument('--dps_code', required=True)
p.add_argument('--config', default='des_y3_like/des_like.yaml')
p.add_argument('--work', default='work')
p.add_argument('--out', default='des_y3_like/results/validate_ia_model.json')
p.add_argument('--seed', type=int, default=7)
a = p.parse_args()
sys.path.insert(0, a.dps_code)
from device_utils import to_device                 # noqa: E402
from ia_forward import IAForwardModel              # noqa: E402
from MapTools import TorchMapTools                 # noqa: E402

cfg = yaml.safe_load(open(a.config))
info = json.load(open(os.path.join(a.work, 'maps_info.json')))
n, theta, nb = cfg['map']['n_grid'], cfg['map']['theta_max'], cfg['map']['n_tomo']
A_true = float(cfg['ia']['A_IA'])
sp = np.load(cfg['ia']['spectra_file'])
ia = IAForwardModel(sp['ell'], sp['C_kk'], sp['C_Ik'], sp['C_II'], n, theta, device='cpu', dtype=torch.float64)

dx = np.radians(theta) / n
A_pix = dx ** 2
l1 = 2 * np.pi * np.fft.fftfreq(n, d=dx)
ell = np.hypot(l1[:, None], l1[None, :])
edges = np.geomspace(60, np.pi / dx, 8)
idx = np.digitize(ell, edges)
lc = np.sqrt(edges[1:] * edges[:-1])
nl = len(lc)
on_grid = lambda C: np.interp(np.log(np.clip(ell, sp['ell'][0], sp['ell'][-1])), np.log(sp['ell']), C)
binned = lambda g: np.array([g[idx == i].mean() for i in range(1, nl + 1)])


def cross(x, y):
    pw = (np.fft.fft2(x, norm='ortho') * np.conj(np.fft.fft2(y, norm='ortho'))).real * A_pix
    return binned(pw)


th = {k: np.array([binned(on_grid(sp[k][:, b, b])) for b in range(nb)]) for k in ('C_kk', 'C_Ik', 'C_II')}
S_th = np.array([binned(ia.S1[b, b].numpy()) for b in range(nb)])
# theory fraction of the IA variance that R kappa predicts, summed over the modes of the grid
pf_th = np.array([1 - ia.S1[b, b].numpy()[1:, 1:].sum() / on_grid(sp['C_II'][:, b, b])[1:, 1:].sum() for b in range(nb)])
pf_th = np.array([1 - (ia.S1[b, b].numpy().sum() - ia.S1[b, b].numpy()[0, 0]) /
                  (on_grid(sp['C_II'][:, b, b]).sum() - on_grid(sp['C_II'][:, b, b])[0, 0]) for b in range(nb)])

# shear tools on the run device (float32 on MPS, so the amplitude test is done there)
tool = TorchMapTools(n, theta)
k2s = lambda k: torch.stack(tool.do_fwd_KS_batch(k), dim=1)
sigma = cfg['data']['sigma_e'] / np.sqrt(cfg['data']['neff'] * (theta * 60 / n) ** 2)
torch.manual_seed(a.seed)

acc = {k: np.zeros((nb, nl)) for k in ('kk', 'Ik', 'II', 'res')}
var_ia, var_res, corr_res, corr_pred = np.zeros(nb), np.zeros(nb), np.zeros(nb), np.zeros(nb)
A_hat, A_sd, A_clean, pulls = [], [], [], []
n_test = info['n_test']
for j in range(n_test):
    kappa = np.load(os.path.join(a.work, 'test', f'kappa_{j}.npy')).astype(np.float64)
    kia = np.load(os.path.join(a.work, 'test', f'ia_{j}.npy')).astype(np.float64)
    pred = ia.predict(torch.tensor(kappa[None]))[0].numpy()
    res = kia - pred
    for b in range(nb):
        acc['kk'][b] += cross(kappa[b], kappa[b]) / n_test
        acc['Ik'][b] += cross(kia[b], kappa[b]) / n_test
        acc['II'][b] += cross(kia[b], kia[b]) / n_test
        acc['res'][b] += cross(res[b], res[b]) / n_test
        z = lambda m: m - m.mean()
        var_ia[b] += z(kia[b]).var() / n_test
        var_res[b] += z(res[b]).var() / n_test
        corr_res[b] += np.corrcoef(res[b].ravel(), kappa[b].ravel())[0, 1] / n_test
        corr_pred[b] += np.corrcoef(pred[b].ravel(), kia[b].ravel())[0, 1] / n_test
    # amplitude from noisy shear, given the true kappa
    kt = to_device(kappa.astype(np.float32))
    shear = k2s(to_device((kappa + A_true * kia).astype(np.float32)))            # (nb, 2, n, n)
    clean = shear
    shear = clean + sigma * torch.randn_like(clean)
    ones = torch.ones((nb, n, n), device=shear.device)
    ia_dev = ia_dev if j else IAForwardModel(sp['ell'], sp['C_kk'], sp['C_Ik'], sp['C_II'], n, theta, device=shear.device)
    m, s = ia_dev.conditional_A(kt, shear, sigma ** 2 * ones, ones, k2s)
    A_hat.append(m)
    A_sd.append(s)
    A_clean.append(ia_dev.conditional_A(kt, clean, sigma ** 2 * ones, ones, k2s)[0])
    for _ in range(10):
        noisy = clean + sigma * torch.randn_like(clean)
        pulls.append((ia_dev.conditional_A(kt, noisy, sigma ** 2 * ones, ones, k2s)[0] - A_true) / s)

A_hat, A_sd = np.array(A_hat), np.array(A_sd)
out = dict(
    n_test=n_test, A_injected=A_true, sigma_pixel=float(sigma), ell_bins=lc.tolist(),
    spectra_ratio=dict(C_kk=(acc['kk'] / th['C_kk']).tolist(), C_Ik=(acc['Ik'] / th['C_Ik']).tolist(),
                       C_II=(acc['II'] / th['C_II']).tolist()),
    residual_power_ratio=(acc['res'] / S_th).tolist(),
    predicted_fraction_measured=(1 - var_res / var_ia).tolist(), predicted_fraction_theory=pf_th.tolist(),
    corr_prediction_with_true_IA=corr_pred.tolist(), corr_residual_with_kappa=corr_res.tolist(),
    rms_ratio_IA_over_kappa=(np.sqrt(var_ia) / np.array(info['kappa_rms'])).tolist(),
    A_hat=A_hat.tolist(), A_sd=A_sd.tolist(),
    A_mean=float(A_hat.mean()), A_sem=float(A_hat.std(ddof=1) / np.sqrt(n_test)),
    A_scatter=float(A_hat.std(ddof=1)), A_reported_sd=float(A_sd.mean()),
    A_noiseless_mean=float(np.mean(A_clean)), A_noiseless_scatter=float(np.std(A_clean, ddof=1)),
    pull_std=float(np.std(pulls)), pull_mean=float(np.mean(pulls)), n_pulls=len(pulls),
    negative_modes_clipped=int(ia.n_negative_modes))
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(out, open(a.out, 'w'), indent=1)

np.set_printoptions(precision=3, suppress=True, linewidth=150)
print(f'{n_test} held-out maps, {nb} bins, grid {n}^2;  clipped negative-eigenvalue modes: {ia.n_negative_modes}')
print('ell bins:', lc.round())
for k in ('C_kk', 'C_Ik', 'C_II'):
    print(f'measured / theory {k}, bins 1..{nb} (rows):'); print(np.array(out['spectra_ratio'][k]))
print('residual power / S(l):'); print(np.array(out['residual_power_ratio']))
print('fraction of IA variance predicted by R kappa: measured', np.array(out['predicted_fraction_measured']), ' theory', pf_th)
print('corr(R kappa, true kappa_IA):', corr_pred, '   corr(residual, kappa):', corr_res)
print('rms(kappa_IA at A=1) / rms(kappa):', np.array(out['rms_ratio_IA_over_kappa']))
print(f"A_IA given the true kappa over {n_test} noisy mocks: mean {out['A_mean']:.3f} +- {out['A_sem']:.3f} "
      f"(injected {A_true}); scatter {out['A_scatter']:.3f}, reported sd {out['A_reported_sd']:.3f}")
print(f"without noise: A = {out['A_noiseless_mean']:.4f}, scatter {out['A_noiseless_scatter']:.4f};  "
      f"pull over {len(pulls)} noise draws: mean {out['pull_mean']:.3f}, width {out['pull_std']:.3f}")
