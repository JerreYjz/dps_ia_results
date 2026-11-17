"""
Summary statistics of the DPS runs of run_des_like.py (IA modelled against IA ignored).

Per mock and tomographic bin b, with t the true kappa, m the posterior mean (all mean-subtracted):
    corr      correlation of m with t
    slope     <m t> / <t t>: the response of the posterior mean to the true signal
    leak      <D I> / <I I>, with D = m_ignored - m_modelled and I = A * kappa_IA (the injected IA):
              how much of the injected IA shows up in the map when the model ignores it
and for each run the IA amplitude the data prefer given a kappa map (IAForwardModel.conditional_A),
evaluated on the true kappa and on every posterior sample.

Usage (from the top of this repository, in the Cocoa environment):
    DPS_DEVICE=cpu python3 des_y3_like/analyze_dps.py --dps_code <.../denoising_diffusion_pytorch>
"""
import argparse
import glob
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
p.add_argument('--out', default='des_y3_like/results/dps_ia.json')
a = p.parse_args()
sys.path.insert(0, a.dps_code)
from device_utils import to_device                 # noqa: E402
from ia_forward import IAForwardModel              # noqa: E402
from MapTools import TorchMapTools                 # noqa: E402

cfg = yaml.safe_load(open(a.config))
n, theta, nb = cfg['map']['n_grid'], cfg['map']['theta_max'], cfg['map']['n_tomo']
sp = np.load(cfg['ia']['spectra_file'])
tool = TorchMapTools(n, theta)
k2s = lambda k: torch.stack(tool.do_fwd_KS_batch(k), dim=1)
ia = IAForwardModel(sp['ell'], sp['C_kk'], sp['C_Ik'], sp['C_II'], n, theta, device=tool.ell.device)

zm = lambda x: x - x.mean(axis=(-2, -1), keepdims=True)
corr = lambda x, y: float((zm(x) * zm(y)).sum() / np.sqrt((zm(x) ** 2).sum() * (zm(y) ** 2).sum()))
slope = lambda m, t: float((zm(m) * zm(t)).sum() / (zm(t) ** 2).sum())

files = sorted(glob.glob(os.path.join(a.work, 'samples', 'mock_*.npz')),
               key=lambda f: int(os.path.basename(f)[5:-4]))
rows = []
for f in files:
    d = np.load(f)
    t, kia, A = d['kappa'], d['kappa_IA'], float(d['A_injected'])
    shear = to_device(d['noisy_shear'])
    var = to_device(np.full((nb, n, n), float(d['sigma_pixel']) ** 2, dtype=np.float32))
    ones = torch.ones_like(var)
    cond = lambda k: ia.conditional_A(to_device(np.asarray(k, dtype=np.float32)), shear, var, ones, k2s)
    A_truth, A_sd = cond(t)
    row = dict(mock=int(os.path.basename(f)[5:-4]), A_injected=A, n_samples=int(d['modelled'].shape[0]),
               A_given_truth=A_truth, A_sd=A_sd,
               corr_KS=[corr(d['KS'][b], t[b]) for b in range(nb)])
    means = {}
    for run in ('modelled', 'ignored'):
        s = d[run]
        m = means[run] = s.mean(0)
        As = [cond(k)[0] for k in s]
        row[run] = dict(corr=[corr(m[b], t[b]) for b in range(nb)],
                        slope=[slope(m[b], t[b]) for b in range(nb)],
                        rms_sample=[float(zm(s[:, b]).std()) for b in range(nb)],
                        A_given_samples=float(np.mean(As)), A_given_samples_scatter=float(np.std(As)))
    D = means['ignored'] - means['modelled']
    row['leak'] = [slope(D[b], A * kia[b]) for b in range(nb)]
    row['corr_shift_with_IA'] = [corr(D[b], A * kia[b]) for b in range(nb)]
    row['rms_shift_over_rms_mean'] = [float(zm(D[b]).std() / zm(means['modelled'][b]).std()) for b in range(nb)]
    row['rms_truth'] = [float(zm(t[b]).std()) for b in range(nb)]
    row['rms_IA_injected'] = [float(zm(A * kia[b]).std()) for b in range(nb)]
    rows.append(row)

stack = lambda get: np.array([get(r) for r in rows])
mean_sem = lambda x: (x.mean(0).tolist(), (x.std(0, ddof=1) / np.sqrt(len(x))).tolist() if len(x) > 1 else None)
sm, si = stack(lambda r: r['modelled']['slope']), stack(lambda r: r['ignored']['slope'])
summary = dict(
    n_mocks=len(rows), n_bins=nb,
    slope_modelled=mean_sem(sm), slope_ignored=mean_sem(si),
    slope_ratio_minus_1=mean_sem(si / sm - 1),
    corr_modelled=mean_sem(stack(lambda r: r['modelled']['corr'])),
    corr_ignored=mean_sem(stack(lambda r: r['ignored']['corr'])),
    corr_KS=mean_sem(stack(lambda r: r['corr_KS'])),
    leak=mean_sem(stack(lambda r: r['leak'])),
    leak_over_slope=mean_sem(stack(lambda r: r['leak']) / sm),
    corr_shift_with_IA=mean_sem(stack(lambda r: r['corr_shift_with_IA'])),
    rms_shift_over_rms_mean=mean_sem(stack(lambda r: r['rms_shift_over_rms_mean'])),
    A_given_truth=mean_sem(stack(lambda r: [r['A_given_truth']])),
    A_given_modelled=mean_sem(stack(lambda r: [r['modelled']['A_given_samples']])),
    A_given_ignored=mean_sem(stack(lambda r: [r['ignored']['A_given_samples']])),
    A_sd_per_patch=float(np.mean([r['A_sd'] for r in rows])))
prior = os.path.join(a.work, 'samples', 'prior.npz')
if os.path.exists(prior) and rows:
    pk = np.load(prior)['kappa']
    summary['prior_rms_over_truth_rms'] = (zm(pk).std(axis=(0, 2, 3)) / np.mean([r['rms_truth'] for r in rows], axis=0)).tolist()

os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(dict(about='analyze_dps.py on work/samples/mock_*.npz', summary=summary, mocks=rows), open(a.out, 'w'), indent=1)

np.set_printoptions(precision=3, suppress=True, linewidth=150)
print(f"{len(rows)} mocks, A injected = {rows[0]['A_injected'] if rows else None}")
for k in ('corr_KS', 'corr_modelled', 'corr_ignored', 'slope_modelled', 'slope_ignored', 'slope_ratio_minus_1',
          'leak', 'leak_over_slope', 'corr_shift_with_IA', 'rms_shift_over_rms_mean', 'A_given_truth', 'A_given_modelled', 'A_given_ignored'):
    m, s = summary[k]
    print(f'{k:22s} mean {np.array(m)}   sem {np.array(s) if s is not None else None}')
print('prior rms / truth rms:', np.array(summary.get('prior_rms_over_truth_rms', [])))
