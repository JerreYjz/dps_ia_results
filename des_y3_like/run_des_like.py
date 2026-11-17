"""
DPS posterior sampling on DES Y3-like mocks that contain shell-built NLA IA, with the IA
forward model switched on and off.

For each held-out map j the mock data are
    shear = KS[ kappa_j + A * kappa_IA_j(shells) ] + shape noise          (A = ia: A_IA in the config)
and two posteriors are sampled from the same data with the same random seed:
    'modelled'  forward model kappa + A R(l) kappa      (ia_forward.py)
    'ignored'   forward model kappa                     (the original code)
Same seed means the same starting noise and the same noise at every step, so the difference
between the two sets of samples is the effect of the IA term, not sampling scatter.

Outputs: work/samples/mock_{j}.npz with the truth, the IA map, the data, the KS map and the
samples (n_samples, n_bins, npix, npix) of both runs; work/samples/prior.npz with prior samples.

Usage (from the top of this repository, in the Cocoa environment, after training):
    python3 des_y3_like/run_des_like.py --dps_code <.../denoising_diffusion_pytorch> --mocks 0 1 2 3
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import yaml

p = argparse.ArgumentParser()
p.add_argument('--dps_code', required=True)
p.add_argument('--config', default='des_y3_like/des_like.yaml')
p.add_argument('--work', default='work')
p.add_argument('--mocks', type=int, nargs='+', default=[0, 1, 2, 3])
p.add_argument('--n_samples', type=int, default=4)
p.add_argument('--model', default='1', help='checkpoint number in train: results_folder')
p.add_argument('--guidance', default='dps_sigmoid')
p.add_argument('--delta_t', type=float, default=500.)
p.add_argument('--sigma_t', type=float, default=20.)
p.add_argument('--steps', type=int, default=1000)
p.add_argument('--n_prior', type=int, default=8)
a = p.parse_args()
sys.path.insert(0, a.dps_code)
from device_utils import DEVICE, to_device                               # noqa: E402
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer  # noqa: E402
from ia_forward import IAForwardModel                                    # noqa: E402
from utils import string_to_numpy_array                                  # noqa: E402

cfg = yaml.safe_load(open(a.config))
n, theta, nb = cfg['map']['n_grid'], cfg['map']['theta_max'], cfg['map']['n_tomo']
A_true = float(cfg['ia']['A_IA'])
kmin = string_to_numpy_array(cfg['train']['transform']['kappa_min'])[:, None, None]
kmax = string_to_numpy_array(cfg['train']['transform']['kappa_max'])[:, None, None]
sigma = cfg['data']['sigma_e'] / np.sqrt(cfg['data']['neff'] * (theta * 60 / n) ** 2)
sigma_noise = to_device(sigma * np.ones((nb, n, n), dtype=np.float32))
mask = to_device(np.ones((nb, n, n), dtype=np.float32))

model = Unet(dim=64, flash_attn=False, channels=nb).to(DEVICE)
diffusion = GaussianDiffusion(model, image_size=n, theta_max=theta, timesteps=1000, sampling_timesteps=a.steps,
                              noisy_image=None, sigma_noise=sigma_noise, survey_mask=mask,
                              kappa_min=kmin, kappa_max=kmax, ddim_sampling_eta=1.).to(DEVICE)
trainer = Trainer(diffusion, cfg['train']['data_folder'], calculate_fid=False,
                  results_folder=cfg['train']['results_folder'])
trainer.load(a.model)
sp = np.load(cfg['ia']['spectra_file'])
ia = IAForwardModel(sp['ell'], sp['C_kk'], sp['C_Ik'], sp['C_II'], n, theta, device=DEVICE)
diffusion.ia_model = ia
if a.guidance.startswith('dps_sigmoid'):
    diffusion.set_dps_lkl_scale(np.full(nb, a.delta_t), np.full(nb, a.sigma_t))


def unnorm(x):
    return x.detach().cpu().numpy() * (kmax - kmin) + kmin


out_dir = os.path.join(a.work, 'samples')
os.makedirs(out_dir, exist_ok=True)

if a.n_prior and not os.path.exists(os.path.join(out_dir, 'prior.npz')):
    torch.manual_seed(1)
    np.savez(os.path.join(out_dir, 'prior.npz'), kappa=unnorm(diffusion.sample(batch_size=a.n_prior)))

for j in a.mocks:
    t0 = time.time()
    kappa = to_device(np.load(os.path.join(a.work, 'test', f'kappa_{j}.npy')))
    kia = to_device(np.load(os.path.join(a.work, 'test', f'ia_{j}.npy')))
    diffusion.A_IA = 0.0                                                   # plain KS for the mock data
    torch.manual_seed(1000 + j)
    clean = torch.stack(diffusion.kappa_to_shears((kappa + A_true * kia)[None]))[:, 0]     # (nb, 2, n, n)
    noisy = clean + sigma * torch.randn_like(clean)
    ks = torch.stack([diffusion.map_tool.do_KS_inversion(noisy[b]) for b in range(nb)])
    diffusion.noisy_image = noisy
    res = {}
    for name, A_model in (('modelled', A_true), ('ignored', 0.0)):
        diffusion.A_IA = A_model
        torch.manual_seed(5000 + j)                                        # same seed for both runs
        x = diffusion.sample_posterior(batch_size=a.n_samples, guidance_method=a.guidance)
        res[name] = unnorm(x).astype(np.float32)
    np.savez(os.path.join(out_dir, f'mock_{j}.npz'), kappa=kappa.cpu().numpy(), kappa_IA=kia.cpu().numpy(),
             noisy_shear=noisy.cpu().numpy(), KS=ks.cpu().numpy(), sigma_pixel=sigma, A_injected=A_true,
             guidance=a.guidance, **res)
    print(f'mock {j}: done in {time.time() - t0:.0f} s', flush=True)
