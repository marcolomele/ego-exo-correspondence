# Data Directory Structure

## Core Directories

- **`raw/`** - Raw videos downloaded using EgoExo4D CLI. See [setup instructions](raw/INSTRUCTIONS_RAW.md) and [EgoExo4D documentation](https://ego4d-data.org/docs/start-here/).

- **`root/`** - Processed data structure for O-MaMa training and inference. See [setup guide](root/INSTRUCTIONS_ROOT.md).

- **`annotations/`** - Relation annotations from EgoExo4D. See [annotation instructions](annotations/INSTRUCTIONS_ANNOTATIONS.md).

## Processed Datasets

- **`casa_gio/`** - Custom hand-made dataset. See [usage guide](casa_gio/INSTRUCTIONS_CASA_GIO.md).

---

## Training Workflows

This repository supports **two training workflows** that use the same sampled frame pairs:

### Workflow A: On-the-Fly Feature Extraction

Features are computed during training (slower, no preprocessing required).

**Steps:**
1. Download and process data
2. Create frame pairs with sampling:
   ```bash
   cd src/scripts
   # Option 1: Sample by ratio (percentage of frames)
   python create_pairs.py --scenario health --train_ratio 0.5 --val_ratio 0.5 --test_ratio 0.5 --seed 42
   
   # Option 2: Sample exact number of pairs (recommended for consistency)
   python create_pairs.py --scenario health --train_pairs 9100 --val_pairs 1950 --test_pairs 1950 --seed 42
   ```
   > **Note:** `create_pairs.py` is the **single source of truth** for frame sampling. Use `--{split}_pairs` for exact counts or `--{split}_ratio` for percentage-based sampling.

3. Train with on-the-fly features:
   ```bash
   cd src/O-MaMa
   python main.py --root ../../data/root --reverse
   ```

### Workflow B: Precomputed Features

Features are extracted once and reused (faster training, requires preprocessing).

**Steps:**
1-2. Same as Workflow A (create sampled frame pairs)

3. Precompute features for sampled pairs:
   ```bash
   cd src/scripts
   python precompute_features_dinov2.py --root ../../data/root --reverse
   python precompute_features_dinov3.py --root ../../data/root --reverse
   ```
   > **Important:** Precompute scripts should NOT do internal sampling. They extract features for pairs already sampled by `create_pairs.py`.

4. Verify extracted features:
   ```bash
   python create_split_from_features.py --root ../../data/root
   ```
   > This documents which takes successfully had features extracted.

5. Train with precomputed features:
   ```bash
   cd src/O-MaMa
   python main_precomputed.py --root ../../data/root --reverse
   ```

### Key Principles

- **Single Source of Truth:** `create_pairs.py` controls frame sampling for both workflows
- **Consistency:** Both workflows use identical frame pairs from `dataset_jsons/*_pairs.json`
- **Flexibility:** Easy to switch between workflows without regenerating data
- **No Data Leakage:** Take-level splits in `split.json` ensure clean train/val/test separation

### Sampling Options

`create_pairs.py` supports multiple sampling strategies:

| Mode | Arguments | Example | Use Case |
|------|-----------|---------|----------|
| **No sampling** | None | `--scenario health` | Use all available pairs |
| **Uniform ratio** | `--sample_ratio` | `--sample_ratio 0.5` | Sample 50% of frames from all splits |
| **Per-split ratio** | `--{split}_ratio` | `--train_ratio 0.3 --val_ratio 0.7 --test_ratio 0.5` | Different percentages per split |
| **Exact counts** | `--{split}_pairs` | `--train_pairs 9100 --val_pairs 1950 --test_pairs 1950` | Precise control (13K total) |

**Recommendation:** Use exact counts (`--{split}_pairs`) for reproducible experiments and fair comparisons across models.