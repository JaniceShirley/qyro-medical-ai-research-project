# Security Policy: QYRO Medical AI

This document details the security practices, model safety principles, and patient privacy commitments for the **QYRO-Medical-AI** research project.

---

## 🔒 1. Patient Data & Privacy Commitment (No PHI)

**Absolute Protection of Personal Health Information (PHI):**
* **No PHI Allowed:** This repository contains **zero** patient identifiers, private health records, or Protected Health Information (PHI). We strictly comply with HIPAA (Health Insurance Portability and Accountability Act) and GDPR (General Data Protection Regulation) data privacy frameworks.
* **Open Public Datasets Only:** All clinical datasets linked or audited in this workspace are derived from publicly available, anonymized academic repositories that have been formally cleared for research use by their original institutional review boards (IRBs).
* **Local Sandboxing:** No patient-uploaded images or diagnostic telemetry from production endpoints are stored or uploaded to this research repository.

---

## 🛡️ 2. Responsible Vulnerability Disclosure

If you discover a security vulnerability, configuration error, or inadvertent exposure of clinical data, please report it immediately through our private disclosure channel.

### Reporting Process:
* **Contact Email:** Please email our medical security team at `security@qyro.ai` (monitored 24/7).
* **Information to Include:**
  * A description of the issue or configuration exposure.
  * Steps to reproduce the issue.
  * Potential impact on research data or models.
* **Response SLA:** We will acknowledge receipt of your report within **24 hours** and provide a resolution plan within **72 hours**.
* **Safe Harbor:** We support responsible, coordinated disclosure. We ask that you do not publish details of the vulnerability publicly until it has been patched and resolved.

---

## 🤖 3. Model Safety & Clinical Guardrails

As a medical AI research project, model safety is integral to our pipeline:
* **Adversarial Robustness:** We regularly evaluate model sensitivity to image perturbations (e.g., camera focus blur, lighting shifts, compression noise) to ensure stable inference in real-world clinic settings.
* **Skin Tone Bias Auditing:** Every production-candidate model undergoes rigorous equity evaluations using skin tone metadata (e.g., Fitzpatrick Skin Type I–VI and Monk Skin Tone scales) to ensure equitable diagnostic accuracy.
* **Deterministic Configuration:** Training runs use locked seeds and strict deterministic parameters to guarantee reproducible model behaviors.
* **No Diagnostic Execution:** The models and weights developed in this project serve as clinical-decision support tools. They are **not** standalone diagnostics. All final medical decisions must be reviewed and made by a licensed healthcare professional.
