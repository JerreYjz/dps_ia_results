"""
Write the data.json of every figure of the note from the analysis outputs.

Reads (relative to the top of the repository):
    des_y3_like/results/validate_ia_model.json     des_y3_like/validate_ia_model.py
    des_y3_like/results/dps_ia.json                des_y3_like/analyze_dps.py
    des_y3_like/results/wiener_reference*.json     des_y3_like/wiener_reference.py
    work/samples/mock_<j>.npz                      des_y3_like/run_des_like.py
    work/ia_spectra_shells.npz, work/maps_info.json   des_y3_like/make_des_like_maps.py
Writes notes/des_y3_like_ia/plots/<name>/data.json for
    ia_model_test, ia_amplitude, dps_response, maps.

Usage: python3 notes/des_y3_like_ia/plots/make_figure_data.py [--map_mock 0]
"""
import argparse
import json
import os

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--map_mock', type=int, default=0, help='mock shown in the maps figure')
a = p.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
res = lambda *x: os.path.join(TOP, *x)
r3 = lambda v: np.round(np.asarray(v, dtype=float), 4).tolist()


def write(name, data):
    os.makedirs(os.path.join(HERE, name), exist_ok=True)
    json.dump(data, open(os.path.join(HERE, name, 'data.json'), 'w'), indent=1)
    print('wrote', name)


v = json.load(open(res('des_y3_like', 'results', 'validate_ia_model.json')))
nb = len(v['predicted_fraction_theory'])

# --- ia_model_test: measured / theory against ell, per bin
write('ia_model_test', dict(
    about='des_y3_like/validate_ia_model.py, 100 held-out DES Y3-like maps (64^2 pixels of 8 arcmin). '
          'ell: geometric centres of the multipole bins. cross_ratio[b]: measured C^{I kappa}_bb / Limber theory. '
          'residual_ratio[b]: power of (kappa_IA - R kappa) of bin b / model S_bb. '
          'predicted_fraction_*: fraction of the IA variance predicted by R kappa. Written by make_figure_data.py.',
    ell=r3(v['ell_bins']), reference=1.0,
    cross_ratio=r3(v['spectra_ratio']['C_Ik']), residual_ratio=r3(v['residual_power_ratio']),
    kk_ratio=r3(v['spectra_ratio']['C_kk']), II_ratio=r3(v['spectra_ratio']['C_II']),
    predicted_fraction_measured=r3(v['predicted_fraction_measured']),
    predicted_fraction_theory=r3(v['predicted_fraction_theory'])))

dps_file = res('des_y3_like', 'results', 'dps_ia.json')
if os.path.exists(dps_file):
    d = json.load(open(dps_file))
    s, mocks = d['summary'], d['mocks']

    # --- ia_amplitude: A_IA preferred by the data given a kappa map
    write('ia_amplitude', dict(
        about='IA amplitude from the exact Gaussian conditional p(A | kappa, data). truth_all: given the true kappa, '
              '100 noisy mocks (validate_ia_model.py). truth / modelled / ignored: the DPS mocks of run_des_like.py, '
              'given the true kappa and given the posterior samples of the two runs (mean over samples per mock; '
              'analyze_dps.py). sd_per_patch: statistical error of one patch. Written by make_figure_data.py.',
        A_injected=v['A_injected'], zero=0.0, sd_per_patch=round(s['A_sd_per_patch'], 4),
        truth_all=r3(v['A_hat']), truth_all_mean=round(v['A_mean'], 4), truth_all_sem=round(v['A_sem'], 4),
        truth=r3([m['A_given_truth'] for m in mocks]),
        modelled=r3([m['modelled']['A_given_samples'] for m in mocks]),
        ignored=r3([m['ignored']['A_given_samples'] for m in mocks])))

    # --- dps_response: response of the posterior mean per bin, and its change when IA is ignored
    sp = np.load(res('work', 'ia_spectra_shells.npz'))
    info = json.load(open(res('work', 'maps_info.json')))
    n, theta = info['npix'], info['theta_max_deg']
    dx = np.radians(theta) / n
    l1 = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    ell = np.hypot(l1[:, None], l1[None, :])
    g = lambda C: np.interp(np.log(np.clip(ell, sp['ell'][0], sp['ell'][-1])), np.log(sp['ell']), C)
    A = v['A_injected']
    # single-bin Kaiser-Squires expectation of the amplitude bias: A * sum C^{I kappa} / sum C^{kappa kappa}
    ks_bias = [A * (g(sp['C_Ik'][:, b, b]).sum() - g(sp['C_Ik'][:, b, b])[0, 0]) /
               (g(sp['C_kk'][:, b, b]).sum() - g(sp['C_kk'][:, b, b])[0, 0]) for b in range(nb)]
    wf = json.load(open(res('des_y3_like', 'results', 'wiener_reference.json')))
    wfd = json.load(open(res('des_y3_like', 'results', 'wiener_reference_deep.json')))
    per = lambda key: r3([[100 * (mi / mm - 1) for mi, mm in zip(m['ignored'][key], m['modelled'][key])] for m in mocks])
    write('dps_response', dict(
        about='des_y3_like/analyze_dps.py. slope: least-squares slope of the DPS posterior mean against the true kappa, '
              'per tomographic bin; *_mean and *_sem over the mocks. bias_percent: 100 * (slope_ignored / slope_modelled - 1) '
              'per mock (rows) and bin. ks_bias_percent: single-bin Kaiser-Squires expectation '
              '100 * A * sum_l C^{I kappa}_bb / sum_l C^{kappa kappa}_bb over the modes of the grid. '
              'wiener_percent: the same change for the tomographic Wiener filter on the 100 held-out maps at the same '
              'noise (wiener_reference.py); wiener_deep_percent: at wiener_deep_neff galaxies per arcmin^2 per bin; '
              'wiener_single_bin_percent: Wiener filter of one bin at a time (not drawn). '
              'leak: slope of (mean_ignored - mean_modelled) against the injected A kappa_IA. Written by make_figure_data.py.',
        bins=list(range(1, nb + 1)), n_mocks=s['n_mocks'], zero=0.0,
        slope_modelled_mean=r3(s['slope_modelled'][0]), slope_modelled_sem=r3(s['slope_modelled'][1]),
        slope_ignored_mean=r3(s['slope_ignored'][0]), slope_ignored_sem=r3(s['slope_ignored'][1]),
        bias_percent=per('slope'),
        bias_percent_mean=r3(100 * np.array(s['slope_ratio_minus_1'][0])),
        bias_percent_sem=r3(100 * np.array(s['slope_ratio_minus_1'][1])),
        ks_bias_percent=r3(100 * np.array(ks_bias)),
        wiener_percent=r3(100 * np.array(wf['slope_ratio_minus_1'][0])),
        wiener_deep_percent=r3(100 * np.array(wfd['slope_ratio_minus_1'][0])), wiener_deep_neff=wfd['neff'],
        wiener_single_bin_percent=r3(100 * np.array(wf['single_bin_slope_ratio_minus_1'][0])),
        leak_mean=r3(s['leak'][0]), leak_sem=r3(s['leak'][1]),
        corr_modelled=r3(s['corr_modelled'][0]), corr_ignored=r3(s['corr_ignored'][0]), corr_KS=r3(s['corr_KS'][0])))

    # --- maps: bin 1 of one mock
    m = np.load(res('work', 'samples', f'mock_{a.map_mock}.npz'))
    r5 = lambda x: np.round(np.asarray(x, dtype=float), 5).tolist()
    b = 0
    write('maps', dict(
        about=f'work/samples/mock_{a.map_mock}.npz (run_des_like.py), tomographic bin {b + 1}, 64^2 pixels of 8 arcmin. '
              'truth: true kappa. mean_modelled: DPS posterior mean with the IA model. shift: posterior mean with IA '
              'ignored minus with IA modelled (same data, same random seed). injected_IA: A * kappa_IA built from the '
              'shells. Written by make_figure_data.py.',
        mock=a.map_mock, bin=b + 1, extent_deg=theta, n_samples=int(m['modelled'].shape[0]),
        truth=r5(m['kappa'][b]), mean_modelled=r5(m['modelled'].mean(0)[b]),
        shift=r5(m['ignored'].mean(0)[b] - m['modelled'].mean(0)[b]),
        injected_IA=r5(float(m['A_injected']) * m['kappa_IA'][b])))
else:
    print('no', dps_file, ': only ia_model_test written')
