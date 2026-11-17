"""
Write numbers.tex, the numbers block of the note, from the analysis outputs.

Every macro gets a comment naming the file and key it comes from. Run from anywhere:
    python3 notes/des_y3_like_ia/make_numbers.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.abspath(os.path.join(HERE, '..', '..'))
V = 'des_y3_like/results/validate_ia_model.json'
D = 'des_y3_like/results/dps_ia.json'
M = 'work/maps_info.json'
T = 'des_y3_like/results/toy_and_checks.json'
W = 'des_y3_like/results/wiener_reference.json'
WD = 'des_y3_like/results/wiener_reference_deep.json'
v = json.load(open(os.path.join(TOP, V)))
info = json.load(open(os.path.join(TOP, M)))
d = json.load(open(os.path.join(TOP, D)))['summary'] if os.path.exists(os.path.join(TOP, D)) else None
t = json.load(open(os.path.join(TOP, T)))
names = ['One', 'Two', 'Three', 'Four']
lines = []


def macro(name, value, source, refresh=True):
    value = str(value)
    if value[:1] in '+-':                       # signed numbers in math mode, for a true minus sign
        value = f'\\ensuremath{{{value}}}'
    lines.append(f'% {source}')
    lines.append(f'\\newcommand{{\\{name}}}{{\\refresh{{{value}}}}}' if refresh else f'\\newcommand{{\\{name}}}{{{value}}}')


lines.append('% Part 1: results (move when the analysis is rerun).')
for b in range(4):
    macro(f'PredFracMeas{names[b]}', f"{100 * v['predicted_fraction_measured'][b]:.1f}", f'{V}, predicted_fraction_measured[{b}] x 100')
    macro(f'PredFracTheory{names[b]}', f"{100 * v['predicted_fraction_theory'][b]:.1f}", f'{V}, predicted_fraction_theory[{b}] x 100')
    macro(f'CorrPred{names[b]}', f"{v['corr_prediction_with_true_IA'][b]:.3f}", f'{V}, corr_prediction_with_true_IA[{b}]')
    macro(f'IAoverKappa{names[b]}', f"{v['rms_ratio_IA_over_kappa'][b]:.2f}", f'{V}, rms_ratio_IA_over_kappa[{b}] (A_IA = 1)')
macro('CorrResMax', f"{max(abs(x) for x in v['corr_residual_with_kappa']):.3f}", f'{V}, max |corr_residual_with_kappa|')
hi = np.array(v['ell_bins']) > 150
dev = lambda key: 100 * max(np.abs(np.array(v['spectra_ratio'][key])[:, hi] - 1).max() for _ in [0])
macro('SpectraDevHighEll', f"{max(dev(k) for k in ('C_kk', 'C_Ik', 'C_II')):.0f}", f'{V}, max |spectra_ratio - 1| x 100 over the three spectra, ell > 150')
macro('SpectraDevLowEll', f"{max(100 * np.abs(np.array(v['spectra_ratio'][k])[:, ~hi] - 1).max() for k in ('C_kk', 'C_Ik', 'C_II')):.0f}", f'{V}, max |spectra_ratio - 1| x 100 over the three spectra, ell < 150')
macro('ResidFracFour', f"{100 * (1 - v['predicted_fraction_theory'][3]):.0f}", f'{V}, (1 - predicted_fraction_theory[3]) x 100')
macro('ResidualDevHighEll', f"{100 * np.abs(np.array(v['residual_power_ratio'])[:, hi] - 1).max():.0f}", f'{V}, max |residual_power_ratio - 1| x 100, ell > 150')
macro('AmpTruthAllMean', f"{v['A_mean']:.2f}", f'{V}, A_mean')
macro('AmpTruthAllSem', f"{v['A_sem']:.2f}", f'{V}, A_sem')
macro('AmpSdPatch', f"{v['A_reported_sd']:.2f}", f'{V}, A_reported_sd')
macro('AmpNoiseless', f"{v['A_noiseless_mean']:.3f}", f'{V}, A_noiseless_mean')
macro('AmpNoiselessScatter', f"{v['A_noiseless_scatter']:.3f}", f'{V}, A_noiseless_scatter')
macro('PullWidth', f"{v['pull_std']:.2f}", f'{V}, pull_std')
if d is not None:
    macro('NMocks', f"{d['n_mocks']}", f'{D}, summary.n_mocks')
    macro('AmpGivenModelled', f"{d['A_given_modelled'][0][0]:.2f}", f'{D}, summary.A_given_modelled mean')
    macro('AmpGivenModelledSem', f"{d['A_given_modelled'][1][0]:.2f}", f'{D}, summary.A_given_modelled sem')
    macro('AmpGivenIgnored', f"{d['A_given_ignored'][0][0]:.2f}", f'{D}, summary.A_given_ignored mean')
    macro('AmpGivenIgnoredSem', f"{d['A_given_ignored'][1][0]:.2f}", f'{D}, summary.A_given_ignored sem')
    macro('AmpGivenTruth', f"{d['A_given_truth'][0][0]:.2f}", f'{D}, summary.A_given_truth mean')
    macro('AmpGivenTruthSem', f"{d['A_given_truth'][1][0]:.2f}", f'{D}, summary.A_given_truth sem')
    for b in range(4):
        macro(f'Bias{names[b]}', f"{100 * d['slope_ratio_minus_1'][0][b]:+.1f}", f'{D}, summary.slope_ratio_minus_1 mean[{b}] x 100')
        macro(f'BiasSem{names[b]}', f"{100 * d['slope_ratio_minus_1'][1][b]:.1f}", f'{D}, summary.slope_ratio_minus_1 sem[{b}] x 100')
        macro(f'SlopeModelled{names[b]}', f"{d['slope_modelled'][0][b]:.2f}", f'{D}, summary.slope_modelled mean[{b}]')
        macro(f'CorrModelled{names[b]}', f"{d['corr_modelled'][0][b]:.2f}", f'{D}, summary.corr_modelled mean[{b}]')
        macro(f'CorrIgnored{names[b]}', f"{d['corr_ignored'][0][b]:.2f}", f'{D}, summary.corr_ignored mean[{b}]')
        macro(f'CorrKS{names[b]}', f"{d['corr_KS'][0][b]:.2f}", f'{D}, summary.corr_KS mean[{b}]')
        macro(f'Leak{names[b]}', f"{d['leak'][0][b]:.2f}", f'{D}, summary.leak mean[{b}]')
        macro(f'LeakOverSlope{names[b]}', f"{d['leak_over_slope'][0][b]:.2f}", f'{D}, summary.leak_over_slope mean[{b}]')
        macro(f'ShiftCorr{names[b]}', f"{d['corr_shift_with_IA'][0][b]:.2f}", f'{D}, summary.corr_shift_with_IA mean[{b}]')
        macro(f'PriorRms{names[b]}', f"{d['prior_rms_over_truth_rms'][b]:.2f}", f'{D}, summary.prior_rms_over_truth_rms[{b}]')
    macro('RmsShiftPercent', f"{100 * np.mean(d['rms_shift_over_rms_mean'][0]):.0f}", f'{D}, summary.rms_shift_over_rms_mean, mean over bins x 100')
    wf, wfd = json.load(open(os.path.join(TOP, W))), json.load(open(os.path.join(TOP, WD)))
    for b in range(4):
        macro(f'WFBias{names[b]}', f"{100 * wf['slope_ratio_minus_1'][0][b]:+.1f}", f'{W}, slope_ratio_minus_1 mean[{b}] x 100')
        macro(f'WFDeepBias{names[b]}', f"{100 * wfd['slope_ratio_minus_1'][0][b]:+.1f}", f'{WD}, slope_ratio_minus_1 mean[{b}] x 100')
        macro(f'WFSingleBias{names[b]}', f"{100 * wf['single_bin_slope_ratio_minus_1'][0][b]:+.1f}", f'{W}, single_bin_slope_ratio_minus_1 mean[{b}] x 100')
        macro(f'WFSlope{names[b]}', f"{wf['slope_modelled'][0][b]:.2f}", f'{W}, slope_modelled mean[{b}]')
        macro(f'WFCorr{names[b]}', f"{wf['corr_modelled'][0][b]:.2f}", f'{W}, corr_modelled mean[{b}]')
    macro('DeepNEff', f"{wfd['neff']:g}", f'{WD}, neff', refresh=False)
    fig = json.load(open(os.path.join(HERE, 'plots', 'dps_response', 'data.json')))
    for b in range(4):
        macro(f'KSBias{names[b]}', f"{fig['ks_bias_percent'][b]:+.1f}", f'notes/des_y3_like_ia/plots/dps_response/data.json, ks_bias_percent[{b}]')
for k, name in (('toy_A_truth', 'ToyAmpTruth'), ('toy_A_truth_sd', 'ToyAmpTruthSd'), ('toy_A_modelled', 'ToyAmpModelled'),
                ('toy_A_ignored', 'ToyAmpIgnored'), ('roman_spurious_max', 'RomanSpuriousMax'),
                ('train_minutes', 'TrainMinutes'), ('sample_minutes_per_run', 'SampleMinutes'),
                ('shells_vs_ccl_des_percent', 'ShellsVsCCLDes'), ('shells_vs_ccl_roman_percent', 'ShellsVsCCLRoman')):
    macro(name, t[k], f'{T}, {k}')

lines.append('% Part 2: inputs (a rerun does not change them).')
c = info['cosmology']
for name, val, src in (
        ('NPix', info['npix'], 'npix'), ('PixArcmin', f"{info['pixel_arcmin']:.0f}", 'pixel_arcmin'),
        ('FieldDeg', f"{info['theta_max_deg']:.2f}", 'theta_max_deg'), ('NShells', info['n_shells'], 'n_shells'),
        ('NShellsShifted', info['n_shells_shifted'], 'n_shells_shifted'), ('ZMax', f"{info['z_max']:.1f}", 'z_max'),
        ('NTrain', info['n_train'], 'n_train'), ('NTest', info['n_test'], 'n_test'),
        ('OmegaC', c['Omega_c'], 'cosmology.Omega_c'), ('OmegaB', c['Omega_b'], 'cosmology.Omega_b'),
        ('Hubble', c['h'], 'cosmology.h'), ('SigmaEight', c['sigma8'], 'cosmology.sigma8'), ('SpectralIndex', c['n_s'], 'cosmology.n_s')):
    macro(name, val, f'{M}, {src}', refresh=False)
for b in range(4):
    macro(f'KappaRms{names[b]}', f"{1e3 * info['kappa_rms'][b]:.1f}", f'{M}, kappa_rms[{b}] x 1000', refresh=False)
macro('ShellVarMax', f"{info['shell_var_delta_max']:.0f}", f'{M}, shell_var_delta_max', refresh=False)
for name, val, src in (('TrainSteps', 2000, 'des_y3_like/des_like.yaml, train.nsteps'),
                       ('TrainBatch', 24, 'DPS_mass_mapping train.py, train_batch_size'),
                       ('UnetWidth', 64, 'DPS_mass_mapping train.py, Unet dim'),
                       ('DeltaT', 500, 'des_y3_like/run_des_like.py, --delta_t default'),
                       ('SigmaT', 20, 'des_y3_like/run_des_like.py, --sigma_t default'),
                       ('SampleSteps', 1000, 'des_y3_like/run_des_like.py, --steps default'),
                       ('NSamples', 4, 'des_y3_like/run_des_like.py, --n_samples default')):
    macro(name, val, src, refresh=False)
macro('AInjected', v['A_injected'], 'des_y3_like/des_like.yaml, ia.A_IA', refresh=False)
macro('SigmaPixel', f"{v['sigma_pixel']:.3f}", f'{V}, sigma_pixel', refresh=False)
macro('NEff', '1.475', 'des_y3_like/des_like.yaml, data.neff', refresh=False)
macro('SigmaE', '0.26', 'des_y3_like/des_like.yaml, data.sigma_e', refresh=False)
open(os.path.join(HERE, 'numbers.tex'), 'w').write('\n'.join(lines) + '\n')
print(f'wrote numbers.tex with {sum(l.startswith(chr(92)) for l in lines)} macros')
