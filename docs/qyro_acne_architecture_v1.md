# QYRO Acne AI Architecture Specification (v1)

This document details the modular medical AI system design for **QYRO Acne v1**. The architecture prioritizes explainability, clinical alignment, scalability, and safety by breaking inference down into modular, sequential stages rather than using a single end-to-end black-box model.

---

## 1. Pipeline Architecture Flow

```
                      +-----------------------------+
                      |    Patient Image Uploads    |
                      |  (Front, L/R Profiles, CU)  |
                      +--------------+--------------+
                                     |
                                     v
                      +--------------+--------------+
                      |  Stage 0: Image Quality Gate|
                      | (Blur, Lighting, Resolution)|
                      +-------+--------------+------+
                              |              |
           Quality Pass       |              | Borderline/Usable Pass
      (Apply Full Confidence) |              | (Apply Confidence Downgrade)
                              v              v
                      +-------+--------------+------+
                      |  Model 1: Lesion Detector   |
                      |          (YOLOv8n)          |
                      +--------------+--------------+
                                     |
                                     v
                      +--------------+--------------+
                      | Model 2: Subtype Classifier |
                      |      (EfficientNet-B0)      |
                      +--------------+--------------+
                                     |
                                     v
                      +--------------+--------------+
                      |   Model 3: Severity Grader   |
                      | (EfficientNet-B0 Ordinal Reg)|
                      +--------------+--------------+
                                     |
                                     v
                      +--------------+--------------+
                      |   Weighted Multi-Angle       |
                      |     Inference Merging        |
                      +--------------+--------------+
                                     |
                                     v
                      +--------------+--------------+
                      |     Clinical Rule Engine    |
                      |  (Stage 4 Override & Fails) |
                      +--------------+--------------+
                                     |
                                     v
                      +--------------+--------------+
                      |  Final Dermatology Report   |
                      +-----------------------------+
```

---

## 2. Stage 0: Image Quality Gate & Confidence Downgrade

The Image Quality Gate filters out images that are clinically unusable.

### 2.1 Algorithmic Quality Filters
1. **Resolution Filter**: Rejects any image with dimensions $< 640 \times 640$ pixels.
2. **Focus/Blur Check**: Laplacian variance is calculated.
   - $\text{Var(Laplacian)} < 20 \rightarrow$ **REJECT** (e.g., `"Image too blurry. Please retake photo."`).
   - $20 \le \text{Var(Laplacian)} < 60 \rightarrow$ **PASS WITH PENALTY** (Usable/Borderline quality).
   - $\text{Var(Laplacian)} \ge 60 \rightarrow$ **PASS WITH FULL CONFIDENCE** (Sharp quality).
3. **Lighting & Exposure Check**:
   - Computes grayscale histogram distribution.
   - **Overexposure**: If $>25\%$ of pixels have a luminance value $> 245$, **REJECT** (e.g., `"Please avoid direct flash or bright background lighting."`).
   - **Underexposure**: If $>30\%$ of pixels have a luminance value $< 20$, **REJECT** (e.g., `"Please retake photo in daylight or well-lit environment."`).
4. **Face Visibility & Crop check**:
   - Performs face localization using a lightweight detector (e.g., OpenCV Haar Cascade or MediaPipe Face Mesh).
   - Rejects profile/frontal views if no face bounding box is found, unless the upload is explicitly tagged as a "close-up affected region".

### 2.2 Confidence Downgrade Modifier
If an image passes the quality gate but is classified as `borderline` or `usable` clinical quality (Laplacian variance $< 60$ or borderline lighting), the system applies a confidence modifier:
* **Usable Quality (Laplacian $[40, 60)$)**: Multiply all model prediction confidence scores (detections, subtypes, severity) by a factor of **0.85**.
* **Borderline Quality (Laplacian $[20, 40)$)**: Multiply all model prediction confidence scores by a factor of **0.70**.
This penalty prevents over-confident predictions on degraded images, ensuring downstream clinical rule fallbacks are triggered.

---

## 3. Model 1: Lesion Detection (YOLOv8n)

### 3.1 Rationale
* **Nano Variant**: 3.2M parameters. Extremely fast inference ($<10$ms on mobile CPU), fully ONNX-compatible for on-device edge execution.
* **Input Resolution**: $640 \times 640$ pixels with letterbox aspect ratio conservation.
* **Outputs**: Class-specific bounding boxes (`[x_min, y_min, x_max, y_max]`), detection confidence scores ($C_d \in [0.1, 1.0]$), and total lesion count.

---

## 4. Model 2: Subtype Classification (EfficientNet-B0)

### 4.1 Rationale & Target Classes
* **Model Backbone**: EfficientNet-B0 (5.3M parameters). High accuracy-to-parameter ratio.
* **Target Classes**:
  1. `open_comedo` (Blackhead)
  2. `closed_comedo` (Whitehead)
  3. `papular` (Inflammatory Papule)
  4. `pustular` (Pustule)
  5. `cystic` (Nodule/Cyst)
  6. `mixed` (Comedonal + Inflammatory lesions)
  7. `scar` (Acne scars)
  8. `infantile` (Pediatric acne variant)
  9. `mechanica` (Friction-induced acne)
* **Data Augmentations**: Random cropping, horizontal flips, Gaussian noise, color jitter, and grid distortions. **Vertical flips are strictly prohibited** to prevent violating natural gravitational anatomical orientation.

---

## 5. Model 3: Severity Grading (EfficientNet-B0 Ordinal Regression)

### 5.1 Ordinal Class Priority
Standard multi-class cross-entropy treats misclassifications equally (e.g., predicting Stage 4 instead of Stage 1 has the same penalty as predicting Stage 2 instead of Stage 1). In medicine, this is unacceptable. We prioritize **Ordinal Regression** to enforce ordering constraints across:
* `stage_1` (Mild)
* `stage_2` (Moderate)
* `stage_3` (Severe)
* `stage_4` (Very Severe)

### 5.2 Training Target Representation
We reformulate classification using $K-1$ binary classification tasks (where $K=4$):
* Task 1: Is severity $>$ Stage 1?
* Task 2: Is severity $>$ Stage 2?
* Task 3: Is severity $>$ Stage 3?
The final predicted stage is determined by summing the thresholded sigmoid activations of these binary tasks:
$$\text{Severity Stage} = 1 + \sum_{i=1}^{3} \mathbb{I}(p_i > 0.5)$$

---

## 6. Multi-Angle Capture Design & Weighted Aggregation

To capture a complete view of the face, QYRO Acne v1 requires a multi-angle patient upload containing four images:
1. **Frontal View** (Weight $W_f = 0.35$)
2. **Left Profile View** (Weight $W_l = 0.25$)
3. **Right Profile View** (Weight $W_r = 0.25$)
4. **Close-Up Affected Region** (Weight $W_{cu} = 0.15$, if uploaded; otherwise redistribution occurs: $W_f = 0.40, W_l = 0.30, W_r = 0.30$).

### 6.1 Lesion Counting Aggregation
Lesions are aggregated by summing profile views while applying a spatial deduplication factor to the frontal overlap region:
$$\text{Total Count} = \text{Count}_{\text{Left}} + \text{Count}_{\text{Right}} + (0.5 \times \text{Count}_{\text{Front}}) + (0.3 \times \text{Count}_{\text{Close-up}})$$

### 6.2 Severity and Subtype Probability Aggregation
The pooled class probabilities are computed as a weighted average of the individual view prediction vectors:
$$\mathbf{P}_{\text{pooled}} = \sum_{a \in \{\text{views}\}} W_a \cdot \mathbf{P}_a$$
where $\mathbf{P}_a$ is the probability vector for view $a$ after applying any image quality confidence downgrades.

---

## 7. Clinical Rule Engine

The rule engine acts as an explainable decision layer on top of raw model outputs, converting probabilities into structured medical feedback.

### 7.1 Stage 4 Emergency Override
If Model 3 predicts `stage_4` (Very Severe) with a confidence $> 60\%$, or if Model 2 predicts `cystic` acne probability $> 50\%$ across any view, the rule engine triggers an immediate **Emergency Referral Override**:
* **Action**: Interrupt standard pipeline reports.
* **Output Message**: `"CRITICAL ALERT: Severe nodulocystic/inflammatory acne detected. There is a high risk of permanent scarring and physical discomfort. We recommend scheduling an urgent consultation with a board-certified dermatologist for systemic therapy evaluation."`

### 7.2 Uncertainty Fallback Rules
To prevent making unreliable recommendations when models are uncertain:
* **High Entropy Fallback**: If the entropy of the pooled subtype probability vector exceeds a threshold ($H(\mathbf{P}_{\text{pooled}}) > 1.8$), or if the highest class confidence is $< 40\%$:
  - **Action**: Suppress specific subtype tags.
  - **Output Message**: `"Mixed/Indeterminate Acne Presentation: Multiple lesion types co-occur with similar densities. A broad-spectrum management approach is suggested."`
* **Conflict Fallback**: If Model 1 detects zero lesions but Model 3 predicts `stage_3` severity:
  - **Action**: Mark as warning conflict.
  - **Output Message**: `"Data Inconsistency: Bounding box count is zero, but severity estimates suggest active inflammatory changes. Please ensure facial images are clear and retry."`

### 7.3 Medical Explanation Templates & Safety Disclaimers
* **Report Template**:
  ```
  QYRO ACNE SCAN REPORT
  =====================
  Total Lesions Detected: {total_lesions}
  Primary Lesion Subtype: {primary_subtype} (Confidence: {subtype_confidence}%)
  Assessed Severity Stage: {severity_stage} (Confidence: {severity_confidence}%)
  
  Clinical Interpretation: {rule_engine_output}
  ```
* **Required Safety Disclaimer**:
  * `"Disclaimer: QYRO Acne v1 is an investigational decision-support tool. It does NOT constitute medical diagnosis, treatment prescription, or clinical advice. All findings should be reviewed and validated by a licensed physician before starting any medical therapy."`

---

## 8. Explicit v1 Limitations

1. **Resolution and Scale Sensitivity**: Models are optimized for $640 \times 640$. Micro-comedones may be missed if image resolution is low or if the smartphone camera distance exceeds 30 cm.
2. **Skin Tone Bias potential**: Fitzpatrick Skin Types V and VI have limited representation in training sets (SCIN only). Prediction confidence may vary on highly pigmented skin.
3. **Lighting & Shade Vulnerability**: Poor home lighting, yellow incandescent bulbs, or side-shadows can lead to false-positive detection of pustules due to specular reflection.
4. **Anatomical Specificity**: The model is trained on facial acne. Back, chest, and neck acne (acne corporis) will experience high error rates.
5. **No Chronological Tracking**: v1 evaluates single points in time. It cannot track lesion resolution or treatment response across sequential scans.
