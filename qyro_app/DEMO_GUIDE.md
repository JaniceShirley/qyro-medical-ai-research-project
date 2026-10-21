# QYRO demo guide: presenting to a doctor

For whoever runs the demo. Read it once the day before. The technical details are in [README.md](README.md).

## The day before (about 20 minutes)

**1. Get the code.**
```bash
git checkout main
git pull
```
(Or check out the `feat/hybrid-demo-mvp` branch if the pull request is not merged yet.)

**2. Put the two model files** (sent separately; they are not on GitHub) into `models/production/`:
- `qyro_subtype_v1_best.pt`
- `qyro_severity_v1_best.pt`

`qyro_acne_v1_best.pt` is already there.

**3. Install.**
- On Windows with an Nvidia GPU, use your **existing training environment**, which already has CUDA PyTorch. Then:
  ```
  pip install -r qyro_app/requirements.txt
  ```
- A fresh environment would get CPU-only PyTorch. That still works, just more slowly.

**4. Groq key.**
- Copy `qyro_app/.env.example` to `qyro_app/.env` and paste **your own** key after `GROQ_API_KEY=`. A free key is at console.groq.com/keys.
- Without a key the demo still works, but shows a standard pre-written explanation instead of the AI-written one.

**5. Start it.**
- Windows: double-click `qyro_app\run.bat`, or run it from a terminal in the repo folder.
- Mac/Linux: `qyro_app/run.sh`

Open http://localhost:8000.

**6. Check that it works.**
- Open http://localhost:8000/api/health. It should show `"detector": true`, `"subtype": true`, `"severity": true` and `"llm": {... "enabled": true}`.
- Run the three demo cases below once.
- Run the tests: `python -m pytest qyro_app/tests -q` (54 should pass).

**7. Prepare 2-3 photos** on the desktop: a close-up of acne, a full-face acne photo, and a non-skin photo such as a landscape.
- Use **test-split images** from the datasets (for example `datasets/acne_v1_original/test/images/`), never training images.
- **Never use a real patient's photo without consent.**

**8. Internet.** The AI explanation needs internet (Groq). If the venue Wi-Fi fails, the app automatically shows the standard explanation, so the demo still works.

## The demo (about 5 minutes)

**Open with this:**
> "This is QYRO, a decision-support prototype. It does not diagnose. It prepares an assessment that a dermatologist reviews. We'd like your feedback on whether it's useful and where it's wrong."

**Case 1: routine acne photo** (questions left on "Not sure")
1. Upload the photo and click **Analyze photo**.
2. Show the **boxes** on the photo: red means confident, orange means uncertain.
3. Show the three cards: **lesion count**, **acne type**, **severity**, each with its confidence.
4. Show **Explanation & next steps**. Click a source tag (G04, G06, ...) to show it comes from NICE, NHS or AAD guidance.
5. Show the **Flags**. Point out that the system says when it is unsure.

**Case 2: red flags**
1. Same photo, set **"Fever or feeling unwell" = Yes**.
2. The banner turns **red: "Seek prompt medical care"**.
3. Explain: *"Safety rules written in code decide this, not the AI. Red-flag symptoms always escalate, even if the photo is unusable."*

**Case 3: wrong photo**
1. Upload a landscape or a document.
2. It is refused: *"This does not look like a photo of skin."* No AI runs, so nothing is made up.

**If asked how the AI text is kept safe:**
> "The language model only sees the measured results and the cited guidance, never the photo. Its answer is checked automatically for invented numbers, wrong citations, medicine names or doses not in the guidance, and diagnosis claims. If any check fails, a standard pre-written explanation is shown instead."

## Numbers you can quote

These are measured on held-out **test** images on 2026-10-21. Use only these.

| Model | Result |
|---|---|
| Lesion detector | Finds about **61%** of marked lesions (mAP@50 **64%** on test). Counts are undercounts. |
| Acne type | **87%** accuracy on close-up images, across 5 main types (open/closed comedo, papular, pustular, cystic) |
| Severity (4 stages) | **70%** exact, **99%** within one stage. **It tends to under-grade stage 3**, so the rules escalate when stage 3/4 is possible. |

## Do NOT say

- That it **diagnoses** acne or replaces a dermatologist.
- **"73.8% accuracy."** That experiment never finished training; see `reports/experiment_1_dataset_expansion.md`.
- That it is **validated, approved or ready for patients.** It has not been tested on this clinic's patients and has no regulatory approval.
- That it works equally well on **all skin tones.** This has not been measured properly (very few darker-skin test images).
- That the **guidance text is approved.** It is a draft summary that needs a dermatologist's sign-off.

## Likely doctor questions

| Question | Honest answer |
|---|---|
| "How accurate is it?" | Use the table above. Then: "Not yet tested on your patients. That is the next step we'd like your help with." |
| "Why does it say 'review soon' so often?" | "It is tuned to not miss severe cases. On test photos it sent 12 of 14 stage-3 and all stage-4 cases to priority review. The cost is extra referrals for mild cases, and we'd like your view on that balance." |
| "Why did it call this 'infantile' / a strange type?" | "The acne-type model was trained on close-up skin photos, so on full-face photos it can be wrong, and it shows a warning. That is a known limitation we're fixing." |
| "Where do the recommendations come from?" | "From 14 short cited summaries of NICE NG198, NHS and AAD guidance. They are drafts, and we'd like a dermatologist to review them." |
| "Is patient data stored or sent anywhere?" | "The app stores nothing. The photo never leaves the computer. Only the measured results are sent to the AI service to write the explanation. Before real use we'll confirm the provider's data terms." |
| "What about darker skin?" | "Not yet measured well. We'd need images from your patients across skin tones." |

## What to ask the doctor (the real goal of the demo)

1. Which mistakes are **dangerous** for you (for example missing severe acne), and which are **acceptable** (an extra referral)?
2. Would your dermatologists **label 200-300 consented, de-identified photos** of your patients so we can test it properly?
3. Can a dermatologist **review the 14 guidance passages** (`qyro_app/knowledge/guidance.json`)?
4. Which **acne types and grading scale** do you actually use? Should rare types be merged or removed?
5. Who handles **patient consent and data rules** (and regulatory questions) on your side?

Write the answers down and share them with the team. They decide the next work.

## If something goes wrong

| Problem | Fix |
|---|---|
| A card says "Not available" | That model file is missing from `models/production/`. Copy it in and restart. |
| The explanation badge says "Standard text · AI explanation off" | No Groq key, no internet, or Groq was slow. The demo still works. Check `qyro_app/.env` and the internet. |
| "groq: API error 404" in the server log | Groq retired the model. Set `QYRO_LLM_MODEL=` to a current model in `qyro_app/.env` (see the README). |
| The page doesn't load | The app isn't running. Start `run.bat` / `run.sh` again and wait for "Application startup complete". |
| The photo is refused as blurry or dark | Expected for a bad photo. Use a sharper, well-lit one. |
| The first analysis is slow | The models load on first use (a few seconds). Do one test analysis before the doctor arrives. |
