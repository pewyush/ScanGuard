# Relevance of Algorithms and Techniques to ScanGuard

*Companion artifact to `review-of-existing-systems.md` and `objectives-and-methodology.md`. Maps each algorithm/technique from the literature onto ScanGuard's stated project plan (README §Tentative Plan), explains *why* it is relevant, and flags where a technique is relevant only as a negative result or a benchmark anchor. Ends with concrete decisions implied by the research.*

---

## 1. The research question and what the algorithms must serve

ScanGuard's research question (README:

> *Can a deep learning model learn the normal visual characteristics of a manufactured product well enough to detect previously unseen defects, without being trained on labeled defect examples?*

This is exactly the **cold-start problem** that every reviewed family solves. The relevance test for each algorithm below is therefore: does it (a) illuminate that question as a **compared method**, (b) provide a **baseline/floor**, or (c) provide **benchmark context** (MVTec AD 2) against which both are judged?

---

## 2. Mapping algorithms to the plan steps

| Plan step (README) | Relevant algorithms/techniques | Why relevant | Direction of relevance |
|---|---|---|---|
| 3. Baseline model (convolutional AE) | AE / VAE, pixel-level reconstruction error; optionally deep feature reconstruction | The project's exact baseline. Literature says this is the **weakest** modern family — AEs over-generalize and reconstruct defects. Demonstrating this on MVTec AD 2 is the project's core negative-result narrative, not a failure to hide. | Compared baseline (floor) |
| 4. Model iteration (stronger baseline) | PaDiM, SPADE, PatchCore (embedding family); optionally SuperAD (training-free DINOv2) | The planned "stronger baseline" maps directly onto this family. PaDiM is the natural first pick (cheap, statistical, no training; on MVTec AD 2 PatchCore-class methods still fall below ~30% on hard categories, so it is a *teachable* comparison rather than a trivially winning SOTA). | Compared method (stronger baseline) |
| 4. Localization from the same model | kNN patch scoring (SPADE/PatchCore), Mahalanobis distance (PaDiM), AU-PRO / pixel-AUROC metrics | Embedding methods give per-pixel anomaly maps *for free*; the AE needs a separate mechanism. This is the concrete axis where the two compared methods differ — a core finding the write-up should surface. | Comparison axis |
| 6. Analysis: failure modes | EfficientAD (student–teacher), DRAEM (discriminative reconstruction) | Both directly *address* the AE's documented failure modes. They are relevant in the **discussion**: they show the failure is a property of the naive baseline's assumption, not of the reconstruction idea per se — but separately as out-of-scope follow-ups (DRAEM needs anomaly synthesis; EfficientAD is a training/compute step up). | Discussion / future work only |
| 2–6. Everything | MVTec AD 2 split design (normal-only train/val, public vs private GT, lighting shifts), AU-PRO(0.05)/AU-PRO(0.30) | The benchmark's design decisions (high normal variance, small defects, transparent/reflective objects, lighting shift) are precisely the settings where the AE over-generalization and the ImageNet domain bias show up. Relevance: choose 1–3 categories that *contrast* a "reasonable" and a "hard" scenario so the failure modes have something to bind to. | Evaluation context |

---

## 3. Why the embedding family is the relevant "stronger baseline"

1. **No per-dataset training** — PaDiM is purely statistical (Gaussian fitting, no backprop); PatchCore is feature extraction + coreset. This keeps the iterated experiment on budget and directly contrasts with the AE's full training loop — the compute comparison is itself a finding.
2. **Localization without extra machinery** — a Mahalanobis/kNN anomaly map is inherent; the write-up can compare localization *at the same cost* as image-level detection.
3. **The benchmark numbers are unresolved** — on MVTec AD 2, even PatchCore (~53.8% AU-PRO) leaves room; this means the comparison can discuss *where* embedding methods break (hard categories below ~30%), rather than merely reproducing a saturated SOTA. That is the project's stated rationale for picking MVTec AD 2.

**Caveat to plan for in the write-up (known domain mismatch):** ImageNet-pretrained backbones are biased toward natural images; on transparent/reflective industrial objects this causes false detections. This is a *relevant limitation* section, not a blocker.

---

## 4. Relevance of the metric layer

- **Image-level:** AUROC/precision-recall/F1 (README step 5). Straightforward; the primary result of the compared methods.
- **Pixel-level:** the paper's own conclusion is that pure pixel AUROC is inflated by extreme class imbalance (tiny defects). **AU-PRO** is the region-scoped, size-fair metric and is what all published MVTec AD 2 numbers are reported in. Therefore: primary localization metric AU-PRO via the official utils; pixel-AUROC as a fallback only if the official utils prove unwieldy (consistent with `feasibility.md`).
- The stricter **AU-PRO(0.05)** used throughout the MVTec AD 2 paper is the honest ceiling: it will make the AE baseline look poor and the embedding methods merely moderate — exactly the demonstration the research question wants.

---

## 5. What the project should *not* adopt (relevance = explicitly out of scope)

| Technique | Why it stays out | Where it belongs |
|---|---|---|
| DRAEM-style anomaly synthesis | Turns unsupervised AD into a synthetic-discriminative setup; needs a defect-synthesis pipeline that would confound the "seen no defects" narrative of the baseline comparison. | Future work / discussion of how reconstruction families are rescued. |
| GAN-based reconstruction (AnoGAN/f-AnoGAN) | Unstable training, heavier compute, same over-generalization ancestor failure; added complexity with no added teaching value over a plain AE baseline. | Mentioned in review, not implemented. |
| EfficientAD (student–teacher + global AE) | Moderate training, more engineering (distillation, anti-mimicry loss, calibration); valuable as a *reported benchmark anchor* (best published average on MVTec AD 2) rather than a reproduced method. | Reported numbers + future work. |
| Normalizing flows (FastFlow/CFLOW) | Density estimation on high-variance normals risks mis-scoring legitimately diverse normals as anomalies — the exact MVTec AD 2 regime; not needed to answer the research question. | Discussion (why density is risky here). |
| SuperAD / DINOv2 training-free | Strong recent result (won VAND 3.0), but with a ViT-L backbone (~300M params) it is heavy, and its *training-free* framing duplicates the embedding-family story. | Optional cheap third point of comparison if GPU allows; otherwise cite as evidence that pretrained features matter. |

---

## 6. Concrete decisions implied by the research

1. **Baseline = plain convolutional AE** on normal-only training, exactly as planned; score by pixel-wise (and optionally feature-wise) reconstruction error.
2. **Stronger baseline = a PaDiM-style method** (pretrained CNN embedding + per-position multivariate Gaussians + Mahalanobis distance), with PatchCore reported as a literature anchor rather than re-implemented, unless extra budget appears.
3. **Categories:** pick a mix — e.g. one "reasonable" category and one that is transparent/reflective or small-defect. Hard categories that push even PatchCore below ~30% confuse the comparison; a mix keeps both failure modes discussable (ref: `feasibility.md` risk 2).
4. **Metrics:** image-level AUROC/PR/F1 as primary, AU-PRO (official utils) as localization primary, pixel-AUROC as fallback (ref: `feasibility.md` fallback 3). Report AU-PRO(0.05) too for honest comparison with published numbers.
5. **Narrative uses the literature's own admissions:** the AE over-generalization problem (DRAEM's motivating observation), the ImageNet domain bias of embeddings, and the saturation of older benchmarks are all *stated in the reviewed papers* — ScanGuard's comparison is a demonstration of those documented failure modes on an unsaturated benchmark, not an invented critique.

---

## 7. One-line summary

The reconstruction baseline and an embedding method are **the** required comparison for the research question; every other technique reviewed is relevant only as a *benchmark anchor* (EfficientAD, PatchCore numbers), a *discussion point* (DRAEM, flows, SuperAD), or an explicit *out-of-scope extension*.

---

## Key references (same as review + objectives docs)

- Heckler-Kram et al. — *The MVTec AD 2 Dataset*. IJCV 134(4), 2026. arXiv:2503.21622.
- Roth et al. — *Towards Total Recall in Industrial Anomaly Detection* (PatchCore). CVPR 2022. arXiv:2106.08265.
- Defard et al. — *PaDiM*. ICPR-W 2021. arXiv:2011.08785.
- Batzner et al. — *EfficientAD*. WACV 2024. arXiv:2303.14535.
- Zavrtanik et al. — *DRAEM*. ICCV 2021. arXiv:2108.07610.
- Zhang et al. — *SuperAD*. VAND 3.0 report. arXiv:2505.19750.