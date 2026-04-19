# Pipeline Leak Detection with Transfer Learning

This repository contains a pipeline leakage detection project built around transfer learning and fine-tuning for leak position classification.

## Overview

- Main training entry: `train.py`
- Fine-tuning entry: `tune_test.py`
- Config files: `config/train_config.json`, `config/tune_config.json`
- Core model exports: `DualPiecesNet`, `SimpleLoraNet`

## Structure

```text
.
├─config/
├─datasets/
├─models/
├─utils/
├─tools/
├─train.py
├─tune_test.py
├─visualize.py
└─requirements.txt
```

## Requirements

Recommended:

- Python 3.10
- PyTorch 1.9+
- CUDA optional but preferred

Install dependencies:

```bash
pip install -r requirements.txt
pip install seaborn
```

## Data Layout

Default dataset path:

```text
./data/leak_signals
```

Expected layout:

```text
data/leak_signals/
├─0/
├─1/
├─2/
└─3/
```

Each domain folder contains class folders `2` to `13`, and each class folder contains CSV files.

Current loader behavior in `datasets/leak_signals_TL.py`:

- reads CSV with header row
- uses columns `[1, 3, 5]`
- keeps the first `1792` time steps
- uses 12 leak position classes

## Training

Run with the default config:

```bash
python train.py
```

Override config values from CLI if needed:

```bash
python train.py --model_name DualPiecesNet --data_dir ./data/leak_signals --cuda_device 0
```

Default training config is in `config/train_config.json`.

## Fine-Tuning

Run:

```bash
python tune_test.py
```

Typical workflow:

1. Train a base model with `train.py`
2. Put the trained weight path into the fine-tuning config
3. Run `tune_test.py`

## Outputs

Training output folders contain files such as:

- `train.log`
- `config.json`
- `*-best_model.pth`

Fine-tuning output folders may contain:

- `tune.log`
- `confusion_matrix_*.png`
- `*-fine_tuned_*.pth`

## Notes

- `train.py` is the recommended training entry.
- `train_leak.py` is an older script with more hard-coded behavior.
- `visualize.py` is only a helper script and does not fully reconstruct training curves by itself.
- The current fine-tuning code has a config naming mismatch: `tune_config.json` uses `model_name` and `model_path`, while `utils/tune_utils.py` reads `model_Fine_name` and `model_Fine`.

## License

See `LICENSE`.
