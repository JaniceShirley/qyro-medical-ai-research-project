# Contributing to QYRO Medical AI

Thank you for your interest in contributing to the **QYRO Medical AI** research project. We welcome collaboration from university researchers, dermatologists, medical imaging experts, and dataset authors. 

As a clinical AI project, we enforce strict guidelines to ensure research reproducibility, model safety, and code quality.

---

## 🔬 1. Research & Clinical Contributions
We welcome contributions in the following research domains:
* **Annotation Refinements:** Providing feedback on bounding box margins or identifying label noise in clinical datasets.
* **Clinical Validation Study Logs:** Sharing external validation protocols or benchmarking results on new patient demographics.
* **Algorithmic Fairness Papers:** Research regarding bias mitigation, particularly across Fitzpatrick and Monk skin tone groups.

---

## 📊 2. Dataset Ingestion Requirements
To maintain compliance and traceability, any proposed dataset contribution must meet the following criteria before integration:
1. **Provenance:** You must provide clear licensing documentation verifying that the dataset is cleared for academic research or commercial usage.
2. **Deduplication:** Images must pass our automated perceptual hashing (dHash) duplicate check to prevent train-test split contamination.
3. **No Image Uploads:** Never commit raw or cleaned medical images (`.jpg`, `.png`, `.npy`) directly to this repository. Only contribute downloader scripts, preprocessing templates, and metadata manifests (`.csv`, `.json`).

---

## 💻 3. Coding Standards (PEP 8)
* **Python Compliance:** Code must follow standard PEP 8 formatting guidelines.
* **Static Typing:** Type hints are highly encouraged for all function signatures and core pipeline APIs.
* **Deterministic Seeding:** Any training or sampling scripts must explicitly expose random seed configurations to ensure reproducible runs.
* **Comments & Docstrings:** All new functions must include clear docstrings explaining clinical parameters, inputs, and outputs.

---

## 🌿 4. Git Conventions & Workflows

### 4.1 Branch Naming Conventions
Follow structured branch prefixing:
* `research/your-topic` — Core machine learning research and architecture exploration.
* `dataset/new-source` — Dataset downloader scripts and configuration yaml integrations.
* `docs/update-name` — Documentation improvements, research reports, and guides.
* `fix/issue-name` — Bug fixes in pipelines, configurations, or verification utilities.

### 4.2 Commit Message Format (Conventional Commits)
We enforce Conventional Commits for clear tracking:
* `feat:` — Introducing a new data pipeline utility, downloader, or config template.
* `fix:` — Correcting a bug in a preprocessing script or registry validator.
* `docs:` — Modifications to reports, readmes, or roadmap files.
* `dataset:` — Registering a new dataset source or updating metadata manifests.
* `chore:` — Dependencies updates, environment configurations, and file cleaning.

Example: `dataset: register Kurnaz YOLOv8 dataset and add sha256 checksums`

---

## 🐛 5. Issue Reporting Guidelines
When filing an issue, please use the following checklist:
1. **Context:** Specify the environment, hardware profile, and dataset version.
2. **Steps to Reproduce:** Provide the exact command line arguments used.
3. **Expected vs. Actual:** Document the configuration discrepancy or pipeline failure.
4. **No PHI:** Ensure that no patient data, screenshots showing private clinic indicators, or unanonymized images are attached.
