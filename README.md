# Towards Exo-Ego Correspondence 👀
## A Technical Review of the State of the Art – Featuring Ego-Exo4D and Object Mask Matching (O-MaMa)

Augmenting object detection and tracking from Ego POV with Exo POV using deep visual correspondence. Research project for Deep Learning for Computer Vision (20060) Bocconi MSc course.

Built on [O-MaMa](https://github.com/Maria-SanVil/O-MaMa) with support for multiple backbone architectures (DINOv2, DINOv3, ResNet-50) and precomputed feature extraction for accelerated training.

<p align="center">
  <img src="assets/ego_exo_banner.png" alt="Ego-Exo Correspondence overview" width="600"/>
</p>


---

## 📁 Project Structure

```
ego-exo-correspondence/
├── config/                    # Environment configuration
│   └── requirements.txt       # Python dependencies
├── data/                      # Dataset storage (gitignored)
│   ├── raw/                   # Raw EgoExo4D videos
│   ├── root/                  # Processed data for O-MaMa
│   ├── casa_gio/              # Custom hand-made dataset
│   └── annotations/           # Relation annotations
├── docs/                      # Project documentation
│   ├── BOTTLENECK_ANALYSIS.md
│   ├── DATA_PIPELINE_GUIDE.md
│   └── RELATION_DATA_GUIDE.md
├── notebooks/                 # Jupyter notebooks
├── results/                   # Experiment outputs
│   ├── training_run_*/        # Training logs & checkpoints
│   ├── evaluation_*_run_*/    # Evaluation metrics
│   └── timing_profile_*/      # Performance benchmarks
└── src/                       # Source code
    ├── O-MaMa/                # Core model implementation
    ├── scripts/               # Data processing & utilities
    ├── fastsam_extraction/    # FastSAM mask extraction
    └── dinov3-main/           # DINOv3 backbone setup
```

---

## ⚡ Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/ego-exo-correspondence.git
cd ego-exo-correspondence

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r config/requirements.txt
```

### 2. Data Preparation

```bash
# Download and process EgoExo4D data
cd src/scripts
python download_and_process_data.py --scenario health

# Generate ego-exo pairs
python create_pairs.py --data_dir ../../data/root --scenario health

# Extract FastSAM masks
cd ../fastsam_extraction
python extract_masks_FastSAM.py
```

### 3. Precompute Features (Recommended)

Precomputing features provides **10x faster training** by caching backbone outputs:

```bash
cd src/scripts

# DINOv3 (default, 384-dim features)
python precompute_features_dinov3.py --root ../../health_normal_data_omama

# DINOv2 (768-dim features)
python precompute_features_dinov2.py --root ../../health_normal_data_omama

# ResNet-50 (2048-dim features)
python precompute_features_resnet50.py --root ../../health_normal_data_omama
```

---

## 🚀 Usage

### Training

```bash
cd src/O-MaMa

# Train with DINOv3 features
python main_precomputed.py \
    --root ../../health_normal_data_omama \
    --reverse \
    --patch_size 16

# Train with DINOv2 features
python main_precomputed.py \
    --root ../../health_normal_data_omama \
    --reverse \
    --patch_size 14 \
    --dino_feat_dim 768

# Train with ResNet-50 features
python main_precomputed.py \
    --root ../../health_normal_data_omama \
    --reverse \
    --dino_feat_dim 2048
```

### Evaluation

```bash
cd src/O-MaMa

# Evaluate trained model
python main_eval_precomputed.py \
    --root ../../health_normal_data_omama \
    --reverse \
    --patch_size 16 \
    --checkpoint_dir train_output/run_XXX/model_weights/best_IoU_run_XXX.pt

# Evaluate baseline (no fine-tuning)
python main_eval_precomputed.py \
    --root ../../health_normal_data_omama \
    --reverse \
    --patch_size 16
```

---

## 📊 Results

Results are organized by experiment type:

| Directory | Contents |
|-----------|----------|
| `training_run_*` | Training logs, loss curves, model checkpoints |
| `evaluation_baseline_run_*` | Baseline (pretrained) model metrics |
| `evaluation_finetuned_run_*` | Fine-tuned model metrics |
| `timing_profile_*` | Performance benchmarks |
| `casa_gio_*` | Custom dataset evaluation |

Each evaluation produces:
- `results_metrics_run_*.json` — Per-sample IoU scores and predictions
- `evaluation_run_*.log` — Aggregate metrics (mean IoU, accuracy)

---

## 🔧 Key Components

| Component | Description |
|-----------|-------------|
| **O-MaMa** | Object Matching with Masked Attention model for correspondence |
| **FastSAM** | Fast Segment Anything for proposal mask extraction |
| **DINOv2/v3** | Self-supervised vision transformers for feature extraction |
| **ResNet-50** | CNN backbone alternative (DINO pretrained) |

---

## 📚 Documentation

- [`docs/DATA_PIPELINE_GUIDE.md`](docs/DATA_PIPELINE_GUIDE.md) — End-to-end data preparation
- [`docs/RELATION_DATA_GUIDE.md`](docs/RELATION_DATA_GUIDE.md) — EgoExo4D annotation format
- [`docs/BOTTLENECK_ANALYSIS.md`](docs/BOTTLENECK_ANALYSIS.md) — Performance optimization notes

---

## 📄 License

This project is for academic purposes. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [O-MaMa](https://github.com/Maria-SanVil/O-MaMa) — Base model architecture
- [EgoExo4D](https://ego4d-data.org/) — Dataset
- [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) — Mask extraction
- [DINOv1](https://github.com/facebookresearch/dino) / [DINOv2](https://github.com/facebookresearch/dinov2) / [DINOv3](https://github.com/facebookresearch/dinov3) — Feature backbones
