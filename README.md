# MolFORM: Preference-Aligned Multimodal Flow Matching for Structure-Based Drug Design

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

MolFORM is a multimodal flow-matching model for **structure-based drug design (SBDD)**, with preference alignment (e.g., DPO) and reward-based fine-tuning (NFT; Vina + SA) to improve generated molecules.

![MolFORM Architecture](./assets/online_rl_model_figure.png)

This repo uses a **Python-only runtime interface**. All tasks are unified under:

```bash
python -m scripts.run <subcommand> --runtime-config <config.yml>
```

## Contents

- [Environment](#environment)
- [Data Preparation](#data-preparation)
- [Quick Start](#quick-start)
- [Unified Commands](#unified-commands)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Environment

We recommend using Conda:

```bash
conda env create -f environment.yml
conda activate molform
```

## Data Preparation

All user-facing paths live in `./configs/runtime/*.yml`. By default, `paths.data_root` points to `./data`.

For most tasks, `data_root` must contain at least:

- `./data/crossdocked_v1.1_rmsd1.0_pocket10/` (dataset directory)
- `./data/crossdocked_pocket10_pose_split.pt` (train/val/test split)
- `./data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb` (processed LMDB)
- `./data/test_set/` (proteins used for evaluation / sampling)
- `./data/crossdocked_v1.1_rmsd1.0_pocket10_pdb/` (protein structures for docking reward; used by NFT)

DPO training additionally needs a preference dataset file (set `inputs.dpo_data` in `./configs/runtime/train_dpo.yml`). This repo includes example files under `./data/dpo_data/`.

Data references:

- Base data reference: [TargetDiff](https://github.com/guanjq/targetdiff)
- DPO data reference: [AliDiff](https://github.com/MinkaiXu/AliDiff)

Data preprocessing helpers are in `./scripts/data_preparation/`.

## Quick Start

This quick start runs sampling + evaluation (requires a GPU and a trained checkpoint).

Checkpoint download: [Google Drive](https://drive.google.com/drive/folders/1B9ZFcNjdBFH5jyjmTM0nthV8B3q4t1--?usp=sharing)
Place downloaded ckpt files under `./ckpt/` (for example `./ckpt/molform_base_model.pt`), then set the corresponding path in runtime configs (such as `inputs.checkpoint` or `inputs.init_checkpoint`).

1) Set `inputs.checkpoint` in `./configs/runtime/sample_nft.yml`, then run:

```bash
python -m scripts.run sample-nft --runtime-config ./configs/runtime/sample_nft.yml
```

2) Set `inputs.sample_path` in `./configs/runtime/eval_simple.yml` to the sampling output directory (under `./outputs_sampling` unless overridden), then run:

```bash
python -m scripts.run eval-simple --runtime-config ./configs/runtime/eval_simple.yml
```

For training (base/DPO/NFT), see [Unified Commands](#unified-commands).

## Unified Commands

Task runtime configs live in `./configs/runtime/`:

- `./configs/runtime/train_base.yml`
- `./configs/runtime/train_dpo.yml`
- `./configs/runtime/train_nft_vina_sa.yml`
- `./configs/runtime/sample_nft.yml`
- `./configs/runtime/eval_simple.yml`

You usually only need to edit **runtime configs**. Keep all project paths as relative paths (for example `./...`). The unified runtime will generate a temporary, patched config under `paths.tmp_root` at launch time.
Legacy template YAMLs are internal runtime inputs; end users should not need to edit non-runtime configs.

### 1) Base Training (`train-base`)

Train the base flow-matching model.

```bash
python -m scripts.run train-base --runtime-config ./configs/runtime/train_base.yml
```

Optional warm start:

- Set `inputs.checkpoint` in `./configs/runtime/train_base.yml`

### 2) DPO Preference Alignment (`train-dpo`)

Fine-tune with Direct Preference Optimization (DPO).

Required in `./configs/runtime/train_dpo.yml`:

- `inputs.dpo_data`: preference data file (e.g., a `.pkl`)
- `inputs.dpo_ref_ckpt`: reference checkpoint (base model)

```bash
python -m scripts.run train-dpo --runtime-config ./configs/runtime/train_dpo.yml
```

### 3) NFT Fine-tuning (Vina + SA) (`train-nft-vina-sa`)

Fine-tune with rewards built from docking score (Vina) and synthetic accessibility (SA).

```bash
python -m scripts.run train-nft-vina-sa --runtime-config ./configs/runtime/train_nft_vina_sa.yml
```

Optional warm start:

- Set `inputs.init_checkpoint` in `./configs/runtime/train_nft_vina_sa.yml`

### 4) Sampling (`sample-nft`)

Generate molecules using a trained checkpoint.

Required in `./configs/runtime/sample_nft.yml`:

- `inputs.checkpoint`: checkpoint file to sample from

```bash
python -m scripts.run sample-nft --runtime-config ./configs/runtime/sample_nft.yml
```

Sampling outputs are written to `paths.output_root` (default: `./outputs_sampling`). You can override the output folder with `inputs.result_path`.

### 5) Evaluation (CPU path) (`eval-simple`)

Evaluate a sampling directory.

Required in `./configs/runtime/eval_simple.yml`:

- `inputs.sample_path`: path to a sampling output directory

```bash
python -m scripts.run eval-simple --runtime-config ./configs/runtime/eval_simple.yml
```

`eval-simple` is the only supported evaluation subcommand in the unified runtime.

## License

MIT. See `LICENSE`.
