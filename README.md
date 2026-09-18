# ScanGuard

**Unsupervised Visual Anomaly Detection for Industrial Quality Control**

---

## Problem Statement

Quality control is an essential part of manufacturing, where identifying defects accurately and consistently is critical for maintaining product quality and reducing production losses. Traditional inspection methods often rely on manual examination or predefined rules, which can be time-consuming, subjective, and difficult to scale. Manufacturing defects also vary considerably in appearance, size, shape, and location — scratches, cracks, dents, holes, surface contamination, structural abnormalities — making them difficult to catch with simple image-processing techniques.

ScanGuard explores a deep learning based approach to automated visual inspection. Rather than training a model to directly classify *defect vs. no defect* — which requires large amounts of labeled defective examples that rarely exist in real manufacturing settings — the project frames the task as **anomaly detection**:

```
Normal product images
        ↓
CNN / Convolutional Autoencoder
        ↓
Learn the distribution of "normal" appearance
        ↓
New product image
        ↓
Anomaly score
        ↓
Defect flag + (optionally) localized region
```

This framing is well-suited to manufacturing, where normal products are abundant but any individual defect type is rare and defect types themselves are diverse and hard to enumerate in advance. The core research question:

> **Can a deep learning model learn the normal visual characteristics of a manufactured product well enough to detect previously unseen defects, without being trained on labeled defect examples?**

This project is scoped around the **AI/ML research question** — model design, training, and evaluation of anomaly detection performance — rather than production deployment, serving infrastructure, or MLOps tooling.

---

## Dataset

**[MVTec AD 2](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2)** — MVTec Software GmbH

MVTec AD 2 is the second-generation successor to the original MVTec AD benchmark, purpose-built because top anomaly detection methods had begun saturating performance on the original dataset. Key characteristics:

- **8 industrial inspection scenarios**, spanning objects and materials not covered in the original MVTec AD (e.g. transparent/reflective surfaces, bulk/overlapping objects, high intra-class variability in "normal" samples, extremely small defects).
- **8,000+ high-resolution images** in total.
- For each scenario: a **training/validation set of defect-free images only** (consistent with the unsupervised anomaly detection setup).
- **Two test splits**, both containing normal and anomalous images under varying lighting conditions:
  - one with **public pixel-precise ground-truth annotations**, usable for local evaluation;
  - one with **non-public ground truth**, evaluable only via MVTec's public benchmark server — useful for an unbiased final check if time allows.
- Licensed under **CC BY-NC-SA 4.0** (non-commercial, academic/research use — fits a coursework project).

This dataset was chosen specifically because it's harder than the original MVTec AD (state-of-the-art methods top out under ~60% average AU-PRO on it), leaving genuine room to demonstrate model design decisions rather than just replicating known benchmark scores.

---

## Tentative Plan

### 1. Problem framing & literature grounding
- Review core approaches to unsupervised visual anomaly detection: reconstruction-based (autoencoders, GANs), embedding/feature-based (PatchCore, PaDiM-style methods), and student-teacher/distillation approaches.
- Decide on 1–3 scenario categories from MVTec AD 2 to start with (rather than all 8), balancing diversity of defect type with compute/time constraints.

### 2. Data pipeline
- Load and explore the chosen category/categories: inspect normal vs. defective samples, image resolution, lighting variation.
- Build preprocessing pipeline (resizing, normalization, augmentation for the normal-only training set).

### 3. Baseline model
- Train a **convolutional autoencoder** on normal images only.
- Anomaly score derived from reconstruction error (pixel-wise or feature-wise).
- Establish baseline classification metrics (image-level: normal vs. anomalous).

### 4. Model iteration
- Compare against a stronger baseline (e.g. a pretrained CNN feature extractor + distance-based anomaly scoring, in the spirit of PaDiM/PatchCore) to see how far a purely reconstruction-based approach falls short.
- Explore whether localization (pixel-level anomaly maps) can be produced from the same model or requires a separate mechanism.

### 5. Evaluation
- **Image-level**: AUROC, precision/recall, F1 for normal vs. anomalous classification.
- **Pixel-level (localization)**: AU-PRO / pixel-AUROC against the ground-truth segmentation masks, where available.
- Qualitative evaluation: visualize anomaly heatmaps against ground-truth masks for a sample of defective images.
- Compare results across the chosen categories to discuss where the approach succeeds/struggles (e.g. small defects, transparent surfaces).

### 6. Analysis & write-up
- Discuss failure modes and what they reveal about the limitations of reconstruction-based anomaly detection.
- Discuss how results compare to published MVTec AD 2 benchmark numbers.
- Explicitly scope out deployment considerations (serving, latency, monitoring) as future work — this project's contribution is the modeling and evaluation, not shipping a production system.

---

## Explicitly Out of Scope
- Model deployment / serving infrastructure
- Real-time inference pipelines
- MLOps tooling (CI/CD, monitoring, drift detection)
- Production-grade explainability dashboards

These may be natural extensions in a follow-up project, but are not part of this project's evaluation criteria.
