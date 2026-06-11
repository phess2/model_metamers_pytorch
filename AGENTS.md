# AGENTS.md

## Cursor Cloud specific instructions

### Product overview

**Model Metamers / metamMuon** is an offline Python research toolkit (Feather et al. 2023, *Nature Neuroscience*). The `refactor_src` branch adds a refactored `src/` package (Lightning training, unified model registry, `scripts/generate_metamers.py`). There is no web server or database — workflows are CLI scripts and notebooks from `/workspace`.

### Primary conda environment: `metamMuon`

Use the refactored stack on branch `refactor_src` (merged here):

```bash
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate metamMuon
cd /workspace
```

Environment file: `yml_files_for_dependencies/03_02_2026_environment.yml` (Python 3.12, PyTorch 2.9.1+cu130).

**First-time install** (conda create may fail on pip; finish manually):

```bash
sed '/^prefix:/d' yml_files_for_dependencies/03_02_2026_environment.yml > /tmp/metamMuon_env.yml
conda env create -f /tmp/metamMuon_env.yml || true
conda activate metamMuon
pip install "git+https://github.com/jenellefeather/chcochleagram.git"
awk '/^  - pip:$/,0' yml_files_for_dependencies/03_02_2026_environment.yml | tail -n +2 | sed 's/^      - //' | grep -v '^chcochleagram' | grep -v '^prefix:' > /tmp/metamMuon_pip.txt
pip install -r /tmp/metamMuon_pip.txt --extra-index-url https://download.pytorch.org/whl/cu130
pip install timm transformers huggingface_hub safetensors
```

Legacy Feather 2023 scripts still use `model_metamers_pytorch_updates_for_new_model_requirements` (Python 3.8) from `feather_metamers_conda_2023_updated_for_new_model_requirements.yml`.

### Large file downloads

Checkpoints and stimulus assets are **not** in git (~15 GB). Download once per VM:

```bash
python download_large_files.py
```

Metamer generation needs at minimum the `assets/` folder (~366 MB).

### GPU vs CPU

- Metamer generation and `RUN_AUDIO_MODEL_GPU_TESTS=1` tests expect an **NVIDIA GPU** (≥11 GB VRAM).
- This Cloud VM may be CPU-only. Refactored unit tests under `tests/test_*.py` (except legacy `test_public_metamers_repo.py`) run on CPU.
- Legacy `build_network.py` files call `model.cuda()`; use CPU patches documented in the old README flow if needed.

### Running tests

**Refactored tests (metamMuon, recommended):**

```bash
conda activate metamMuon
cd /workspace
python -m unittest \
  tests.test_robust_lips_bounds \
  tests.test_audio_classification \
  tests.test_audio_filtering \
  tests.test_robust_vision_loading \
  tests.test_audio_adversarial
```

**Legacy Feather 2023 tests** (old env + symlink):

```bash
conda activate model_metamers_pytorch_updates_for_new_model_requirements
ln -sfn /workspace/model_analysis_folders /workspace/tests/model_analysis_folders
python -m unittest tests.test_public_metamers_repo
```

### Key commands (refactored)

| Task | Command |
|------|---------|
| Metamer generation CLI | `python scripts/generate_metamers.py --help` |
| Vision training | `python scripts/train_vision.py --help` |
| Audio adversarials | `python scripts/generate_audio_adversarials.py --help` |
| Download data | `python download_large_files.py` |

Run scripts from repo root with `PYTHONPATH=/workspace` (or `export PYTHONPATH=/workspace`) so `from src...` imports resolve in `scripts/`.

### Path warnings (expected)

`analysis_scripts` warns if ImageNet / JSIN training paths are missing. Bundled `assets/` pickles suffice for metamer generation.

### Linting

`ruff` is installed in `metamMuon`. No CI lint step is configured in the repo.
