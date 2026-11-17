# DPS mass mapping with intrinsic alignments: results

Results of tests of the IA-informed forward model for Diffusion Posterior Sampling (DPS) mass
mapping. The sampler code is not here: it is `Supranta/DPS_mass_mapping`, branch `better_dps`,
with the IA patch (`ia_forward.py`, `nla_shell_weights.py`, `make_ia_spectra.py`) on the local
branch `ia-forward-model`.

## Contents

| folder | what it is |
|---|---|
| `notes/des_y3_like_ia/` | the note (`des_y3_like_ia.pdf`), its LaTeX source, its numbers block and one folder per figure |
| `des_y3_like/` | the scripts and config of the DES Y3-like test, and `results/` with the numbers they wrote |

`work/` (maps, the trained checkpoint, the posterior samples) is not committed; the scripts
regenerate it.

## The DES Y3-like test in one paragraph

Convergence maps for the four DES Y3 source bins are built from lognormal lightcone shells, and
NLA intrinsic alignments are built from the same shells with the NLA shell weights, so the IA
maps do not come from the model the sampler uses. A diffusion prior is trained on the
convergence maps. Mock DES Y3-depth shear with IA is then analysed twice with DPS, once with the
IA forward model (the part of IA predicted from the four lensing maps, `R(l) kappa`) and once
without it. The note gives the result and its limits.

## Reproducing it

Run from the top of this repository, inside the Cocoa environment
(`conda activate cocoa; cd ~/cocoa/Cocoa; source start_cocoa.sh`), with
`DPS=~/cocoa/dps_mass_mapping/denoising_diffusion_pytorch`:

```bash
python3 des_y3_like/make_des_like_maps.py --dps_code $DPS \
    --nz $ROOTDIR/projects/desy1xplanck/data/nz_source_Y3_unblinded_02_26_21.txt --out work
# copy the printed kappa_min / kappa_max into des_y3_like/des_like.yaml if they changed
DPS_DEVICE=cpu python3 des_y3_like/validate_ia_model.py --dps_code $DPS    # model test, no sampling
python3 $DPS/train.py des_y3_like/des_like.yaml                            # about 40 min on an M3
python3 des_y3_like/run_des_like.py --dps_code $DPS --mocks 0 1 2 3        # about 10 min per mock
DPS_DEVICE=cpu python3 des_y3_like/analyze_dps.py --dps_code $DPS
python3 notes/des_y3_like_ia/plots/make_figure_data.py
python3 notes/des_y3_like_ia/make_numbers.py
for f in ia_model_test ia_amplitude dps_response maps; do
    python3 notes/des_y3_like_ia/plots/$f/plot_$f.py
done
```

Each figure folder (`data.json`, `plot_<name>.py`, PDF, PNG) redraws on its own.
