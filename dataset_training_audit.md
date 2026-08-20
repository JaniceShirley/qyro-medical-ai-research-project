# QYRO Medical AI - Dataset Training Audit Report

This report presents a forensic audit of the datasets used to train the current production model, `qyro_acne_v1_best.pt`, and addresses data leakage concerns regarding the integration of the ACNE04 v2 dataset.

---

## 1. Production Model Origin

* **Was `qyro_acne_v1_best.pt` trained inside this repository?**
  Yes. The production model is identical to the best checkpoint from the YOLOv8s convergence recovery run.
* **Which training script created it?**
  [train_detection.py](file:///c:/Users/KARTHIK V/OneDrive/Desktop/QYRO-Medical-AI/scripts/training/train_detection.py)
* **Which experiment/run generated it?**
  `yolov8s_qyro_acne_v1_convergence_20260624_150252` under the `experiments/detection/` directory.
* **Date:**
  June 24, 2026 (timestamp: `20260624_150252`).

---

## 2. Datasets Used

The workspace contains five distinct datasets registered in the master registry. Their statistics, roles, and training usages are outlined below:

### 1. Kurnaz (YOLOv8 Acne Dataset)
* **Dataset Path:** [kurnaz_yolov8_cleaned_v1](file:///c:/Users/KARTHIK V/OneDrive/Desktop/QYRO-Medical-AI/datasets/skin/acne/cleaned/kurnaz_yolov8_cleaned_v1) (cloned as [acne_v1_original](file:///c:/Users/KARTHIK V/OneDrive/Desktop/QYRO-Medical-AI/datasets/acne_v1_original))
* **Number of Images:** **520 unique images** (originally 927; 313 perceptual duplicates and 94 blurry images were removed during cleaning).
  * Train: 364 images
  * Validation: 78 images
  * Test: 78 images
* **Annotation Type:** YOLO Bounding Box
* **Classes:** 1 (`Acne`, class index `0`)
* **Usage:** **Actively Used**. This was the primary training, validation, and test dataset for the production detection model (`qyro_acne_v1_best.pt`).

### 2. ACNE04
* **Dataset Path:** [acne04_cleaned_v1](file:///c:/Users/KARTHIK V/OneDrive/Desktop/QYRO-Medical-AI/datasets/skin/acne/cleaned/acne04_cleaned_v1)
* **Number of Images:** **752 unique images** (originally 1406; 521 duplicates and 133 blurry images were removed during cleaning).
  * Train: 524 images
  * Validation: 114 images
  * Test: 114 images
* **Annotation Type:** Severity Stage Classification (Level 0, 1, 2, 3)
* **Classes:** 4 (`Level 0` [Mild/Stage 1], `Level 1` [Moderate/Stage 2], `Level 2` [Severe/Stage 3], `Level 3` [Very Severe/Stage 4])
* **Usage:** **Stored / Used for separate task**. Used to train the separate EfficientNet-B0 severity grading model (`train_severity.py`), but was **not** used to train the production object detection model.

### 3. Tiswan (Acne Dataset Image)
* **Dataset Path:** [tiswan_cleaned_v1](file:///c:/Users/KARTHIK V/OneDrive/Desktop/QYRO-Medical-AI/datasets/skin/acne/cleaned/tiswan_cleaned_v1)
* **Number of Images:** **2,632 unique images** (originally 4,617).
* **Annotation Type:** Folder Classification
* **Classes:** 5 (`Blackheads` [open_comedo], `Whiteheads` [closed_comedo], `Papules` [papular], `Pustules` [pustular], `Cyst` [cystic])
* **Usage:** **Stored / Used for separate task**. Used to train the separate EfficientNet-B0 subtype classification model (`train_subtype.py`), but was **not** used to train the production object detection model.

### 5. DermNet (NZ Acne/Rosacea)
* **Dataset Path:** [dermnet_cleaned_v1](file:///c:/Users/KARTHIK V/OneDrive/Desktop/QYRO-Medical-AI/datasets/skin/acne/cleaned/dermnet_cleaned_v1)
* **Number of Images:** **325 unique images** (originally 1,152).
* **Annotation Type:** Folder Classification
* **Classes:** Subtypes (`open_comedo`, `closed_comedo`, `papular`, `cystic`, `infantile`, `mechanica`, `pustular`, `scar`, etc.)
* **Usage:** **Stored Only**. Maintained in the repository as a clinical reference atlas and external validation anchor for subtype classification. It was **not** used to train the production object detection model.

### 6. Google SCIN (Skin Condition Image Network)
* **Dataset Path:** [acne_subset](file:///c:/Users/KARTHIK V/OneDrive/Desktop/QYRO-Medical-AI/datasets/skin/acne/raw/google_scin/extracted/acne_subset)
* **Number of Images:** **205 unique images**
* **Annotation Type:** Dermatologist Consensus (Fitzpatrick skin type and body region metadata)
* **Classes:** 6 Fitzpatrick Skin Types (`FST1` to `FST6`) and body region labels.
* **Usage:** **Stored Only**. Used exclusively as a robustness holdout evaluation set. It was **not** used to train the production object detection model.

---

## 3. ACNE04 Investigation

* **Was ACNE04 used?**
  * **For the production detector model (`qyro_acne_v1_best.pt`):** **No**. The model is a bounding box detector, whereas the current ACNE04 download lacks bounding box coordinate annotations.
  * **For severity grading:** **Yes**. It was the sole dataset used to train the EfficientNet-B0 severity model.
* **ACNE04 v1 or ACNE04 v2?**
  ACNE04 v1 (the academic release under `cleaned_v1` in the registry).
* **Entire dataset or subset?**
  A cleaned subset (752 unique images out of the original 1,406 raw images).
* **Which YAML or config references it?**
  [severity_config.yaml](file:///c:/Users/KARTHIK V/OneDrive/Desktop/QYRO-Medical-AI/configs/severity_config.yaml) (defining `data_dir: "datasets/skin/acne/final/severity"` and `manifest_path: "datasets/skin/acne/final/metadata/severity_manifest.csv"`).
* **Which training command references it?**
  The command `python scripts/training/train_severity.py --config configs/severity_config.yaml`.

---

## 4. Final Training Dataset

The exact dataset configuration YAML used for the final production training run of `qyro_acne_v1_best.pt` is:

**YAML Path:** [data.yaml](file:///c:/Users/KARTHIK V/OneDrive/Desktop/QYRO-Medical-AI/datasets/acne_v1_original/data.yaml)

```yaml
train: train/images
val: valid/images
test: test/images

nc: 1
names: ['Acne']

clean_reconstruction: true
version: 1.0
reconstruction_date: "2026-06-05"
```

* **train path:** `train/images` (resolves to `datasets/acne_v1_original/train/images` / 364 images)
* **val path:** `valid/images` (resolves to `datasets/acne_v1_original/valid/images` / 78 images)
* **test path:** `test/images` (resolves to `datasets/acne_v1_original/test/images` / 78 images)
* **class names:** `['Acne']`

---

## 5. Data Leakage Assessment

> **SAFE**

### Rationale:
A forensic perceptual hash comparison (dHash) was executed between the 520 unique images of the Kurnaz dataset (which trained `qyro_acne_v1_best.pt`) and the 752 unique images of the ACNE04 dataset. 
* **Result:** **0 perceptual overlaps** were found between the two datasets.
* **Conclusion:** Because the production detector was trained exclusively on the Kurnaz dataset and has zero overlapping images with ACNE04, processing ACNE04 v2 and including it in QYRO Dataset v2 presents no risk of data leakage or split contamination with the current production detector model.

---

## 6. Recommendation

> **Use ACNE04 only as a reference dataset.**

### Rationale:
1. **Licensing Constraints:** The feasibility study (`docs/acne_dataset_research.md` Section 2.1) identifies that ACNE04 is under a non-commercial academic license ("strict academic usage"). Including it in the core training stack violates licensing compliance for commercial deployment.
2. **Missing Bounding Boxes:** The current ACNE04 download lacks bounding box coordinates (`registry/dataset_registry.csv`), making it unsuitable for training object detection models. 
3. **Clinical Benchmarking:** The standardized frontal clinical portraiture makes ACNE04 highly valuable as a calibration and validation benchmark (Production Suitability Score: 8/10 for validation). Keeping it segregated as a test/reference anchor ensures clean clinical evaluations.
