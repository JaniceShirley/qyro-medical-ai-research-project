# QYRO Medical AI - Google SCIN Acne Subset Extraction Report
**Phase 4A: Metadata Investigation & manifest Generation**

---

## 1. Executive Summary

This report documents the forensic metadata audit and case extraction of the acne-related subset of the **Google SCIN (Skin Condition Image Network) Dataset** located in `datasets/skin/acne/raw/google_scin/original_download/`. 

The primary goal of this phase was to verify whether SCIN contains high-quality, clinically labeled acne cases and to construct a download manifest of exact image IDs to fetch, avoiding downloading irrelevant skin conditions (which would waste bandwidth and disk space).

* **Total SCIN Labeled Cases:** 5,033
* **Discovered Acne Cases:** **98**
* **Total Unique Acne Images Extracted:** **205**
* **Fitzpatrick Types Represented:** I–VI (with 62 cases unknown/none identified)
* **Monk Skin Tones (India Scale) Represented:** MST 1 to MST 8 (highest density in MST 2, 3, and 4)
* **Ingestion Recommendation:** **ACCEPT PARTIAL** (Download ONLY the 205 acne-specific images using the generated manifest).

---

## 2. Acne-Related Labels Discovered in SCIN

Our scan of the `scin_labels.csv` file discovered several dermatologist-provided diagnoses containing acne-related terms:

* **`Acne`:** 109 occurrences (the primary target condition).
* **`Folliculitis`:** 297 occurrences (common differential diagnosis).
* **`Rosacea`:** 57 occurrences (common differential diagnosis).
* **`Acne keloidalis`:** 2 occurrences (rare condition).
* **`Acne urticata`:** 1 occurrence.
* **`Acneiform eruption`:** 1 occurrence.
* **`O/E - pustules`:** 4 occurrences.
* **`On examination - follicular pustules`:** 1 occurrence.
* **`O/E - purulent pustules`:** 1 occurrence.

### Extraction Strategy:
To maintain maximum data purity for acne classification, we extracted only the cases where **`Acne`** was listed as a primary diagnosis in the dermatologist-weighted label dictionary (weighted score $\ge 0.3$). This yielded exactly **98 cases** and **205 unique images**.

---

## 3. Metadata Richness

The Google SCIN dataset provides an exceptionally rich set of metadata columns, making it ideal for deep clinical validation:

* **Demographics:** Age group, sex at birth, and self-reported race/ethnicity.
* **Physical Presentation:** Symptoms (itching, burning, pain, bleeding, increasing size, bothersome appearance) and texture characteristics (raised/bumpy, flat, rough/flaky, fluid-filled).
* **Body Location:** Image mapping to specific anatomical areas (e.g. `head_or_neck` represents 52 cases).
* **Capture Profile:** Shot type classifications (`AT_DISTANCE`, `AT_AN_ANGLE`, `CLOSE_UP`).

---

## 4. Smartphone Realism Usefulness

> [!TIP]
> **HIGH REALISM VALUE**
> Because SCIN is a crowdsourced dataset where images were uploaded by real users in their homes using consumer smartphones, it provides an invaluable validation benchmark for "in-the-wild" testing:
> * **mobile Camera Diversity:** Captures variance in lens compression, sensor noise, and digital zoom.
> * **Real-World Illumination:** Includes warm home light bulbs, shadows, mixed window light, and camera flash glare.
> * **Selfie Variations:** Captures realistic portrait angles, blur from hand jitter, and soft-focus crops, matching the exact telemetry inputs expected from QYRO application users.

---

## 5. Indian Skin Tone Robustness Usefulness

One of the key values of Google SCIN is the inclusion of **Monk Skin Tone (MST)** scale annotations, including a specific annotation set calibrated for India (`monk_skin_tone_label_india`):

```
MST 1 (Lightest)       ██ 2 cases
MST 2                  ██████████████████████████████ 48 cases
MST 3                  ███████████████████ 30 cases
MST 4                  ██████ 10 cases
MST 5                  ████ 7 cases
MST 8 (Darkest)        █ 1 case
```

* **Indian Demographics Generalization:** MST values 2, 3, 4, and 5 cover the typical skin tone spectrum in the Indian subcontinent (ranging from fair/wheatish to dark brown). SCIN provides **95 unique cases** within this range.
* **Bias Prevention:** Validating our models against this subset allows QYRO to mathematically measure and mitigate algorithmic bias across dark skin tones before deployment in diverse clinical markets.

---

## 6. Manifest Generation and Next Step

We generated the image download manifest file at **[scin_acne_image_manifest.txt](file:///c:/Users/KARTHIK V/OneDrive/Desktop/QYRO-Medical-AI/docs/scin_acne_image_manifest.txt)**. 

### Recommendation: ACCEPT PARTIAL
QYRO should ingest the SCIN dataset by executing a download script that reads the manifest file and downloads **only the 205 specified images** from Google's public cloud buckets. The remaining ~9,800 images in SCIN representing non-acne conditions should be ignored.
