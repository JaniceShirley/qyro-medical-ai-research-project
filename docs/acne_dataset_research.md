# QYRO Medical AI - Acne Dataset Feasibility Study
**Phase 1A: Acne Dataset Research & Acquisition Planning**

---

## 1. Executive Summary

To transition from the QYRO-Acne prototype into a production-grade clinical AI research model, we require a robust, diverse, and well-annotated dataset stack. 
- **Target Cleaned Images:** 5,000–6,000 high-quality images.
- **Estimated Raw Ingestion Target:** 8,000+ images.

This feasibility study evaluates **seven (7) free/open-source dermatological datasets** containing acne images. By combining classification-focused and object-detection-focused datasets, we can build a two-stage computer vision pipeline: 
1. **Lesion Detection & Counting** (Object Detection: YOLO/Bounding Box).
2. **Lesion Classification & Grading** (Multi-class Classification: Cysts, Papules, Pustules, Blackheads, Whiteheads).

We have identified a total raw pool of **8,725 images**, which we estimate will yield **~6,840 usable images** after passing through deduplication, resolution filtering, blur checking, and annotation verification.

---

## 2. Detailed Dataset Evaluations

### 1. ACNE04 (Original & v2)
* **Dataset Name:** ACNE04
* **Source URL:** [xpwu95/LDL GitHub](https://github.com/xpwu95/LDL) / [openface-io/acne-lds GitHub](https://github.com/openface-io/acne-lds)
* **License / commercial usability:** Unknown / Non-commercial research only (strict academic usage).
* **Dataset Purpose:** Joint acne severity grading and lesion counting (facial clinical images).
* **Approximate Acne Image Count:** ~1,457 images (original) / ~1,204 images (v2).
* **Annotation Format:** Bounding Box (center coordinates + radii) and image-level severity classification (0–3 scale).
* **Skin Tone Diversity:** Low (primarily East Asian subjects in controlled clinical settings).
* **Image Quality Assessment:** High (clear lighting, high resolution, standardized frontal and profile facial images).
* **Production Suitability Score:** 7/10
* **Pros:** Standardized clinical shots, dense lesion-level bounding boxes, expert-vetted severity grades.
* **Cons:** Licensing ambiguity; low demographic skin tone diversity; restricted to facial regions.
* **Overlap Risk with Other Datasets:** Low (proprietary to the ICCV 2019 paper authors, though sometimes re-uploaded to Kaggle/Roboflow).
* **Recommended Usage in QYRO:** Baseline training for the bounding box object detection (lesion counting) pipeline.
* **Expected Usable Image Count after Cleaning:** ~1,200 images.

---

### 2. Google SCIN (Skin Condition Image Network)
* **Dataset Name:** SCIN
* **Source URL:** [google-research-datasets/scin GitHub](https://github.com/google-research-datasets/scin) / [HuggingFace SCIN](https://huggingface.co/datasets/google/scin)
* **License / commercial usability:** SCIN Data Use License (Permissive for research, but contains strict privacy/anti-re-identification clauses; requires legal review for commercial production use).
* **Dataset Purpose:** Multi-racial, crowdsourced dermatology representation (real-world smartphone images).
* **Approximate Acne Image Count:** 109 images.
* **Annotation Format:** Multi-label classification (1–3 dermatologist labels with confidence scores) and user-reported metadata (age, sex, Fitzpatrick skin type, Monk skin tone). No bounding boxes.
* **Skin Tone Diversity:** Excellent (specifically balanced across Fitzpatrick skin type scales I–VI and Monk skin tones).
* **Image Quality Assessment:** Moderate to Low (crowdsourced smartphone pictures with variable lighting, blur, and resolution).
* **Production Suitability Score:** 6/10
* **Pros:** Unmatched demographic/skin tone diversity; dermatologist-labeled; rich metadata.
* **Cons:** Extremely low sample size for acne (109 images); classification-only; highly variable image quality.
* **Overlap Risk with Other Datasets:** Zero (highly unique crowdsourced collection).
* **Recommended Usage in QYRO:** Diversity validation and bias assessment subset; testing out-of-distribution robustness.
* **Expected Usable Image Count after Cleaning:** ~90 images.

---

### 3. Fitzpatrick17K
* **Dataset Name:** Fitzpatrick17k
* **Source URL:** [colorandheritage/Fitzpatrick17k GitHub](https://github.com/colorandheritage/Fitzpatrick17k)
* **License / commercial usability:** CC BY-NC 4.0 (strictly non-commercial).
* **Dataset Purpose:** Clinical algorithmic equity benchmarking (clinical atlas images).
* **Approximate Acne Image Count:** ~380 images (labeled as "acne vulgaris").
* **Annotation Format:** Classification (disease label, Fitzpatrick skin types I–VI, high-level inflammatory classification). No bounding boxes.
* **Skin Tone Diversity:** Excellent (explicitly balanced across Fitzpatrick scales I–VI).
* **Image Quality Assessment:** Good (standardized clinical atlas images from DermIS and Derm101).
* **Production Suitability Score:** 6.5/10
* **Pros:** Expert-labeled skin tones; high-quality clinical reference photography.
* **Cons:** Restricted by CC BY-NC 4.0; no bounding boxes; relatively small subset size for acne.
* **Overlap Risk with Other Datasets:** High (shares images with original DermIS/Derm101 online atlases).
* **Recommended Usage in QYRO:** Fine-tuning skin-tone classification layers and evaluating model equity/bias across dark skin tones.
* **Expected Usable Image Count after Cleaning:** ~350 images.

---

### 4. DermNet (Kaggle Mirror)
* **Dataset Name:** DermNet
* **Source URL:** [shmalian/dermnet Kaggle](https://www.kaggle.com/datasets/shmalian/dermnet)
* **License / commercial usability:** Restricted/Copyrighted by DermNet NZ (public Kaggle mirror is unlicensed; commercial usage is strictly prohibited without direct license acquisition).
* **Dataset Purpose:** Clinical education atlas (clinical dermatology photography).
* **Approximate Acne Image Count:** ~312 images (grouped under "Acne and Rosacea").
* **Annotation Format:** Multi-class classification. No bounding boxes.
* **Skin Tone Diversity:** Low to Moderate (predominantly lighter Fitzpatrick types typical of western clinical reference sources).
* **Image Quality Assessment:** High (macro lenses, professional medical photography).
* **Production Suitability Score:** 5/10
* **Pros:** Highly detailed clinical presentations of lesions; high resolution.
* **Cons:** Severe licensing/copyright barriers; low demographic diversity; contains mixed rosacea cases.
* **Overlap Risk with Other Datasets:** High (frequently scraped into other aggregate Kaggle/Roboflow sets).
* **Recommended Usage in QYRO:** External validation set to evaluate feature extraction accuracy on pure clinical atlas images.
* **Expected Usable Image Count after Cleaning:** ~250 images.

---

### 5. Acne Dataset Image (by Tiswan)
* **Dataset Name:** Acne Dataset Image
* **Source URL:** [tiswan/acne-dataset-image Kaggle](https://www.kaggle.com/datasets/tiswan/acne-dataset-image)
* **License / commercial usability:** Apache 2.0 (fully permissive for commercial and research use).
* **Dataset Purpose:** Fine-grained acne type classification (cropped/zoomed consumer-grade images).
* **Approximate Acne Image Count:** 4,620 images.
* **Annotation Format:** Classification by folder structure (`Blackheads`, `Whiteheads`, `Papules`, `Pustules`, `Cysts`).
* **Skin Tone Diversity:** Medium (primarily Southeast/East Asian skin tones).
* **Image Quality Assessment:** Moderate (consumer smartphone close-ups, variable lighting and zoom levels).
* **Production Suitability Score:** 8/10
* **Pros:** Large volume (4,620 images); fine-grained category labels; Apache 2.0 license.
* **Cons:** No bounding boxes (classification-only); variable resolution and smartphone noise; requires quality curation.
* **Overlap Risk with Other Datasets:** Medium (derived from public web scraping and localized collections).
* **Recommended Usage in QYRO:** Core classification training backbone; a subset should be manually annotated with bounding boxes for fine-grained lesion detection.
* **Expected Usable Image Count after Cleaning:** ~3,500 images.

---

### 6. Acne Dataset in YOLOv8 Format (by kurnazosman)
* **Dataset Name:** Acne Dataset in YOLOv8 Format
* **Source URL:** [kurnazosman/acne-dataset-in-yolov8-format Kaggle](https://www.kaggle.com/datasets/kurnazosman/acne-dataset-in-yolov8-format)
* **License / commercial usability:** Apache 2.0 (fully permissive).
* **Dataset Purpose:** Bounding-box-based object detection for acne lesions.
* **Approximate Acne Image Count:** 927 images.
* **Annotation Format:** Bounding Box (YOLO txt format).
* **Skin Tone Diversity:** Low to Medium.
* **Image Quality Assessment:** Good (focused crops of face regions with active acne).
* **Production Suitability Score:** 7.5/10
* **Pros:** Pre-annotated bounding boxes; Apache 2.0 license.
* **Cons:** Medium dataset size; annotations require verification for label noise; contains background noise.
* **Overlap Risk with Other Datasets:** High (source images overlap heavily with community Roboflow exports).
* **Recommended Usage in QYRO:** Integrate directly into the training partition for the lesion localization/object detection model.
* **Expected Usable Image Count after Cleaning:** ~750 images.

---

### 7. DeepLearning Ensemble for Automated Acne Detection
* **Dataset Name:** DeepLearning Ensemble for Automated Acne Detection
* **Source URL:** [alexanderb14/deeplearning-ensemble-for-automated-acne-detection Kaggle](https://www.kaggle.com/datasets/alexanderb14/deeplearning-ensemble-for-automated-acne-detection)
* **License / commercial usability:** Custom/Unknown (hosted on Kaggle, requires research-only assumption).
* **Dataset Purpose:** Acne lesion detection and severity count.
* **Approximate Acne Image Count:** 920 images.
* **Annotation Format:** Bounding Box (Pascal VOC xml format, containing 2,847 labeled lesions).
* **Skin Tone Diversity:** Low to Medium.
* **Image Quality Assessment:** Good (standardized face shots, cropped regions).
* **Production Suitability Score:** 6.5/10
* **Pros:** Pre-annotated bounding boxes; high lesion density.
* **Cons:** Licensing ambiguity; requires translation to YOLO/COCO coordinates; possible overlap with ACNE04.
* **Overlap Risk with Other Datasets:** High (shares structural similarity with other public academic datasets).
* **Recommended Usage in QYRO:** Bounding box detection training candidate.
* **Expected Usable Image Count after Cleaning:** ~700 images.

---

## 3. Final Recommended Acne Dataset Stack

To achieve an optimal balance between **commercially compliant training data** and **academic benchmark reference sets**, QYRO will partition its data into three distinct layers:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        QYRO Acne Dataset Stack                         │
└────────────────────────────────────────────────────────────────────────┘
                                    │
    ┌───────────────────────────────┼───────────────────────────────┐
    ▼                               ▼                               ▼
┌───────────────────────┐       ┌───────────────────────┐       ┌───────────────────────┐
│  Core Training Stack  │       │ Benchmark & Dev Stack │       │ Diversity & QA Stack  │
│   (Apache 2.0 Only)   │       │   (Academic / NC)     │       │   (High Diversity)    │
├───────────────────────┤       ├───────────────────────┤       ├───────────────────────┤
│ • Tiswan (3,500)      │       │ • ACNE04 (1,200)      │       │ • Google SCIN (90)    │
│ • Kurnaz YOLOv8 (750) │       │ • DL Ensemble (700)   │       │ • Fitzpatrick17k (350)│
│                       │       │ • DermNet (250)       │       │                       │
└───────────────────────┘       └───────────────────────┘       └───────────────────────┘
```

1. **Core Training Stack (Commercially Permissive):**
   * **Datasets:** Tiswan + Kurnaz YOLOv8.
   * **Usable Count:** ~4,250 images.
   * **License:** Apache 2.0.
   * **Role:** Primary weights training for feature extraction and object detection.

2. **Academic Benchmark & Dev Stack:**
   * **Datasets:** ACNE04 + DeepLearning Ensemble + DermNet.
   * **Usable Count:** ~2,150 images.
   * **License:** Academic / Restricted.
   * **Role:** Validation, baseline comparison, and internal prototyping.

3. **Diversity & Algorithmic Bias Stack:**
   * **Datasets:** Google SCIN + Fitzpatrick17k.
   * **Usable Count:** ~440 images.
   * **License:** Custom Research / CC BY-NC.
   * **Role:** Testing model equity and generalization across Fitzpatrick Skin Types I–VI and Monk Skin Tones.

---

## 4. Acquisition & Download Priority

We recommend downloading and staging datasets in the following order to build the pipeline iteratively:

| Priority | Dataset | Category | Rationale |
| :--- | :--- | :--- | :--- |
| **1** | **Acne Dataset in YOLOv8 Format (Kurnaz)** | Core Train | Quickest path to establish the bounding-box object detection training script due to pre-existing YOLO labels under Apache 2.0. |
| **2** | **Acne Dataset Image (Tiswan)** | Core Train | Establishes the multi-class classification backbone. High image volume enables testing our deduplication and resizing scripts. |
| **3** | **ACNE04** | Benchmark | Academic gold-standard for severity grading and lesion counting. Calibrates model counting performance. |
| **4** | **Fitzpatrick17k** | Diversity | Integrates skin-tone tags into the metadata schemas to prepare for fairness metrics. |
| **5** | **Google SCIN** | Diversity | Evaluates how well the trained models handle low-quality, out-of-focus crowdsourced phone images across diverse groups. |
| **6** | **DeepLearning Ensemble** | Benchmark | Expands the bounding-box testing set. |
| **7** | **DermNet** | Validation | Restricted clinical validation mirror to test clinical feature extraction bounds. |

---

## 5. Risks and Limitations

1. **Licensing Segregation:** Production deployments must strictly exclude models trained on `CC BY-NC 4.0` (Fitzpatrick17k) and `DermNet` images. Our training script architecture must support modular data-inclusions (e.g., config switches to enable/disable specific datasets during build runs).
2. **Annotation Domain Mismatch:** Tiswan (classification) and ACNE04 (bounding box center-radius) use different annotation formats. QYRO needs a centralized parser to convert all bounding boxes into standard COCO/YOLO formats.
3. **Severe Demographics Bias:** Apart from SCIN and Fitzpatrick17k, the larger datasets (ACNE04, Tiswan, Kurnaz) have severe population biases (predominantly East/Southeast Asian). If deployed in Western or African markets without clinical telemetry additions, models will suffer performance degradation.
4. **Resolution and Lighting Noise:** Crowdsourced images (Tiswan, SCIN) suffer from variable lighting. We must implement automated preprocessing pipelines (e.g., histogram equalization or color temperature normalization) to prevent overfitting.
