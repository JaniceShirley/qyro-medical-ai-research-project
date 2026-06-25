# QYRO Medical AI — Repository Structure

This document outlines the professional directory layout for the QYRO project, designed for scalability, reproducibility, and clear separation of datasets and models.

```text
QYRO-Medical-AI/
├── configs/                # YAML configuration files for models and environments
├── datasets/
│   ├── acne_v1_original/   # FROZEN dataset used for Production v1 detector
│   ├── acne_v2_curated/    # ACTIVE workspace for annotation, filtering, and additions
│   │   ├── review/         # Candidate ignore manifests (images pending deletion)
│   │   └── ignored/        # Images formally removed from the dataset
│   ├── hair/               # Hair analysis datasets (Future)
│   └── skin/               # Other skin condition datasets (Future)
│
├── models/
│   ├── production/         # FROZEN best.pt checkpoints and config YAMLs
│   └── archive/            # Historical checkpoints from older experiments
│
├── reports/                # Output analytics, diffs, metrics, and CSV audits
│
├── docs/                   
│   ├── dataset_versions.md     # Frozen dataset baseline record
│   ├── dataset_changelog.md    # Active log of all annotations/images added or removed
│   ├── model_registry.md       # Catalog of trained models and their status
│   ├── repository_structure.md # This document
│   └── deployment.md           # Instructions for deploying the model
│
├── scripts/
│   ├── training/           # Python scripts for kicking off YOLOv8 runs
│   ├── annotation/         # Auto-annotation and labeling scripts
│   ├── dataset/            # Dataset cloning, auditing, and processing scripts
│   └── archive/            # Retired, historical, or one-off scratch scripts
│
└── experiments/            # Active training run logs (Tensorboard, raw weights)
```
