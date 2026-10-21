# QYRO demo app

AI-assisted acne assessment prototype for **dermatologist review**. It does not diagnose.

A photo and a short questionnaire go in. The app checks the photo, runs three vision models,
applies tested safety rules, picks matching guidance from a small cited library, and has an LLM
(Groq or Gemini) explain the result in plain language. The LLM's answer is checked before it is
shown; if the check fails, a safe pre-written explanation is used instead.

```
photo + questionnaire
  -> 1. quality gate      (skin visible, sharpness, brightness, resolution)
  -> 2. vision models     (YOLOv8s lesions, EfficientNet-B0 acne type, EfficientNet-B0 severity)
  -> 4. rules             (uncertainty, red flags, recommendation)
  -> 5. retrieval         (cited guidance passages matching the findings)
  -> 6. LLM explanation   (findings + passages only, never the photo)
  -> 7. safety validator  (pass -> LLM text, fail -> template)
  -> 8. result page
```
Layer numbers follow the architecture in the QYRO handoff document.

## Run it

Everything runs from the repository root.

1. **Python environment** (first time only):
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -r qyro_app/requirements.txt
   ```
2. **Model weights.** These are git-ignored, so get them from a teammate and put them in `models/production/`:
   - `qyro_acne_v1_best.pt` (detector, already in the repo)
   - `qyro_subtype_v1_best.pt`
   - `qyro_severity_v1_best.pt`

   Any missing model shows as "Not available" in the app. It never shows a made-up value.
3. **LLM key (optional).** Copy `qyro_app/.env.example` to `qyro_app/.env` and paste **one** key.
   Without a key, the app uses the pre-written explanation.
4. **Start:**
   ```bash
   qyro_app/run.sh
   ```
   On Windows, double-click `qyro_app\run.bat` instead. Then open http://localhost:8000.

**Presenting the demo?** See [DEMO_GUIDE.md](DEMO_GUIDE.md): setup checklist, demo script, quotable numbers, what not to claim, and questions to ask the doctor.

Other commands:

| What | Command |
|---|---|
| Run the tests | `.venv/bin/python -m pytest qyro_app/tests -q` |
| UI without models (fake outputs, labeled "MOCK MODE") | `QYRO_VISION_MODE=mock qyro_app/run.sh` |
| Check status, models and LLM | `curl localhost:8000/api/health` |
| Analyze a photo from the terminal | `curl -F "image=@face.jpg;type=image/jpeg" -F 'questionnaire={"painful_lumps": true}' localhost:8000/api/v1/analyze` |
| Score a classifier on its test split | `.venv/bin/python scripts/evaluation/evaluate_classifiers_test.py --task severity --checkpoint models/production/qyro_severity_v1_best.pt` |

## Configuration

Set in `qyro_app/.env` or the environment.

| Variable | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` / `GEMINI_API_KEY` | none | LLM key. Whichever is set is used (Groq first). |
| `QYRO_LLM_PROVIDER` | `auto` | `groq`, `gemini` or `none` (forces the template). |
| `QYRO_LLM_MODEL` | `openai/gpt-oss-120b` (Groq), `gemini-2.5-flash` (Gemini) | Provider model. Groq retires models, so check `Groq().models.list()` if you get a 404. |
| `QYRO_LLM_TIMEOUT_SECONDS` | `20` | After this, the template is used. |
| `QYRO_VISION_MODE` | `real` | `mock` returns fixed fake outputs for UI work and tests. |

All thresholds live in [config.py](config.py) and [rules.py](rules.py).

## API

`POST /api/v1/analyze` (multipart)
- `image`: JPG, PNG or WebP, up to 10 MB
- `questionnaire`: JSON. Every field is optional, and anything missing is treated as unknown, never as "no":
  `age_years`, `duration` (`unknown`, `under_1_month`, `1_to_6_months`, `over_6_months`),
  `painful_lumps`, `fever_or_unwell`, `rapid_spread`, `scarring`, `current_treatment`

The response (schema `qyro.findings.v1`, see [schemas.py](schemas.py)) contains:
- `image_quality`: each check with pass / caution / reject
- `vision_findings`: each model's name, version, availability, boxes or probabilities
- `safety_flags`
- `recommendation`
- `summary`
- `retrieved_evidence`
- `explanation`: LLM or template text, provider, model, validation result, token counts
- `latency_ms`

## How the recommendation is decided

Fixed Python rules decide the recommendation; the models and the LLM never do. Every rule is covered by tests.

| Recommendation | When |
|---|---|
| `seek_prompt_medical_care` | Fever or feeling unwell is reported. This applies even if the photo is unusable. |
| `retake_photo` | The photo fails the quality gate: too blurry, too dark or bright, too small, or not a skin photo. No models run. |
| `priority_dermatologist_review` | Any of: painful lumps, scarring, rapid spread, age under 12, a cystic pattern, severity stage 3 or 4, **or any uncertain AI result** (see below). |
| `routine_dermatologist_review` | None of the above. |

An **uncertain AI result** is any of:
- low detector confidence
- low acne-type confidence, or an acne type with too little training data
- low severity confidence
- a 25% or higher chance of stage 3 or 4 when stage 1 or 2 is predicted (the severity model tends to under-grade)
- a borderline photo

Uncertain results are labeled "(uncertain)" in the summary.

**Measured tradeoff** (114 severity test photos, no questionnaire answers):

| Rule choice | Stage 3 sent to priority | Stage 4 | Stage 1-2 sent to priority |
|---|---|---|---|
| **Current**: acne-type uncertainty escalates | 12 / 14 | 8 / 8 | 69 / 92 |
| Alternative: acne-type flags only warn | 9 / 14 | 8 / 8 | 28 / 92 |

The current rules favor not missing severe cases. Most extra referrals come from the acne-type
model on full-face photos (see limitations). Choosing between these is a clinical decision.

## LLM safety

The LLM receives only the structured findings, the retrieved guidance and the questionnaire. It never receives the photo.

Its JSON answer is rejected, and the template shown instead, if it:
- states a lesion count or percentage that differs from the findings
- uses any other number that is not in the findings or the guidance
- cites a passage it was not given
- names a medicine that is not in the guidance, or mentions a dose
- claims a diagnosis ("you have cystic acne")
- overclaims ("definitely", "cure")
- echoes injected instructions
- in an urgent case, does not start with same-day care advice

Patient free text and guidance passages are passed as data, and the prompt says to ignore
instructions inside them. Rejected drafts are logged on the server for review.

## Model performance

Measured on held-out **test** splits on 2026-10-21. Details are in `reports/*_test_metrics_20261021.json`.

| Model | Data | Test result |
|---|---|---|
| Detector (YOLOv8s) | Kurnaz, 78 test images | mAP@50 64.1%, precision 65%, recall 61% |
| Acne type (EfficientNet-B0) | Tiswan + DermNet, 453 test images | Accuracy 87.4%. F1: open comedo 0.93, papular 0.89, cystic 0.87, closed comedo 0.85, pustular 0.82. Scar, infantile and mechanica have 1-3 test images, too few to judge. |
| Severity (EfficientNet-B0 ordinal) | ACNE04, 114 test images | Accuracy 70.2%, QWK 0.766, 99.1% within one stage. **Stage 3 recall is only 3/14.** |

## Known limitations

Read these before showing the demo.

- **Prototype only.** It is not validated on the medical center's patients and has no regulatory clearance. Every result needs a dermatologist.
- **Severity test scores are optimistic.** The ACNE04 split puts the same patients in both train and test. A patient-level re-split is needed.
- **The severity model under-grades stage 3.** The rules compensate by escalating, but the model itself needs work.
- **Each classifier only fits one photo type.** The acne-type model learned from close-up crops and is unreliable on full-face photos; it can be confidently wrong, for example "infantile 98%" on an adult face. The severity model learned from full faces. The app warns about both.
- **The detector finds about 61% of lesions,** so counts are undercounts.
- **The skin check is a simple color rule.** It rejects 0.5% of real dataset photos. It was checked on all Fitzpatrick types I-VI, but with only 6 type V and 6 type VI images; the darkest skin tones score lower on average. It can also be fooled by skin-colored objects.
- **The guidance passages** in `knowledge/guidance.json` are paraphrased drafts of NICE NG198, NHS and AAD guidance. **A dermatologist must approve them** before clinical use.
- **Privacy.** On free tiers, Groq and Gemini may keep or use request data. Check their current terms before real patient data is used. The app sends no photos or names, but the free-text "current treatment" box is sent as typed. Nothing is stored by the app, except the server log, which records flags and timings.
- **Training labels are mostly non-expert.** 89% of the acne-type labels are not from dermatologists.
- **Not built yet** (from the handoff): AcneSCU dataset audit, PostgreSQL persistence, Docker Compose, an evaluation summary report.

## Files

| File | Layer |
|---|---|
| [quality.py](quality.py) | 1: quality gate and out-of-scope check |
| [vision.py](vision.py) | 2: model wrapper (preprocessing matches training; tested) |
| [schemas.py](schemas.py) | Request and response contracts |
| [rules.py](rules.py) | 4: flags and recommendation |
| [retrieval.py](retrieval.py), [knowledge/guidance.json](knowledge/guidance.json) | 5: guidance retrieval |
| [llm.py](llm.py) | 6: Groq and Gemini provider wrapper |
| [explain.py](explain.py) | 6-7: prompt, safety validator, template fallback |
| [main.py](main.py), [static/](static/) | API and result page |
| [tests/](tests/) | 52 tests |
