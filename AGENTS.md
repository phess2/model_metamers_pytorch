# AGENTS.md

## Cursor Cloud specific instructions

### Product overview

**Model Metamers** is an offline Python research toolkit (Feather et al. 2023, *Nature Neuroscience*). There is no web server or database. Workflows are CLI Python scripts and Jupyter notebooks run from the repository root (`/workspace`).

### Conda environment

- Miniconda is installed at `/workspace/miniconda3`.
- Activate with:
  ```bash
  source /workspace/miniconda3/etc/profile.d/conda.sh
  conda activate model_metamers_pytorch_updates_for_new_model_requirements
  ```
- The README references `model_metamers_pytorch`; the working env name is `model_metamers_pytorch_updates_for_new_model_requirements` (from `yml_files_for_dependencies/feather_metamers_conda_2023_updated_for_new_model_requirements.yml`). Use the 2022 yml only if the 2023 file fails.
- All commands below assume `cd /workspace` and the activated conda env.

### Large file downloads (required for model tests)

Checkpoints and stimulus assets are **not** in git (~15 GB total). Download once per VM:

```bash
python download_large_files.py
```

Or download only what you need by running selected lines in that script. At minimum, metamer generation needs the `assets/` folder (~366 MB).

### GPU vs CPU

- Metamer generation and full E2E tests expect an **NVIDIA GPU** (≥11 GB VRAM). `build_network.py` files call `model.cuda()`.
- This Cloud VM may be CPU-only. For CPU smoke tests, patch before importing models:
  ```python
  import torch, torch.nn as nn
  _orig_load = torch.load
  torch.load = lambda *a, **k: _orig_load(*a, **{**k, 'map_location': k.get('map_location', 'cpu')})
  nn.Module.cuda = lambda self, device=None: self.to('cpu')
  ```
- Behavioral data analysis, fMRI regression plotting, and notebook figure replication do **not** require a GPU.

### Running tests

From repo root:

```bash
ln -sfn /workspace/model_analysis_folders /workspace/tests/model_analysis_folders  # tests expect this path
python -m unittest tests.test_public_metamers_repo
```

Note: `test_build_networks` iterates all vision/audio models; some (e.g. SWSL) trigger very large Hugging Face / timm downloads. Prefer targeted smoke tests on `alexnet` and `kell2018` when validating setup quickly.

### Key commands

| Task | Command |
|------|---------|
| Install env (first time) | `conda env create -f yml_files_for_dependencies/feather_metamers_conda_2023_updated_for_new_model_requirements.yml` |
| Download data | `python download_large_files.py` |
| Vision metamer (debug) | `python make_metamers_imagenet_16_category_val_400_only_save_metamer_layers.py 0 -I 100 -N 4 -D model_analysis_folders/visual_networks/alexnet` |
| Audio metamer (debug) | `python make_metamers_wsj400_behavior_only_save_metamer_layers.py 0 -I 100 -N 4 -D model_analysis_folders/audio_networks/kell2018` |
| Unit tests | `python -m unittest tests.test_public_metamers_repo` |

### Path warnings (expected)

On startup, `analysis_scripts` prints warnings if ImageNet / JSIN training paths are missing. Metamer generation uses bundled `assets/` pickles and does not require those datasets.

### Linting

No project linter is configured. There is no `Makefile`, `pre-commit`, or CI lint step.
