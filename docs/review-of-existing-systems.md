# Review of Existing Systems

*Delivery artifact for the "Review of Existing Systems & Project Feasibility" check.*

Unsupervised industrial visual anomaly detection is an active research area. The existing work splits into three dominant method families, all of which map onto the approaches staged in the project plan (reconstruction baseline, embedding/feature baseline, and the stated research question).

## 1. Reconstruction-based methods

Train a model to reconstruct *normal* images only; flag anything that reconstructs poorly.

- **Autoencoders / VAEs** (AE, VAE) — the simplest framing: pixel-wise reconstruction error is the anomaly score.
- **GAN-based** — e.g. AnoGAN, f-AnoGAN; a generator/discriminator trained on normals fails on anomalies.
- **Memory-augmented and discriminative variants** — memorised-normal AEs (Gong et al., 2019), and **DRAEM** (Zavrtanik et al., 2021), which adds *synthetic* defects and a discriminative network (needs an anomaly synthesis pipeline).
- **Deep feature reconstruction (DFR)** — reconstructing features of a pretrained network rather than pixels, partially addressing the known weakness below.

**Known weakness (well documented in the literature):** autoencoders can *generalize to anomalies* — they often reconstruct defective regions too well, which is why image-level detection tends to be poor and why this family is usually the weakest modern baseline. This is exactly the risk for the baseline in plan step 3, and it is a legitimate thing to demonstrate rather than hide.

## 2. Embedding / representation-based methods

Extract patch-level features, model the distribution of normal features, and score by distance to that distribution. No per-dataset training.

- **SPADE** — per-pixel nearest-neighbor distance from pretrained features.
- **PaDiM** (Defard et al., 2021) — models patch embeddings as multivariate Gaussians (per-pixel), exploiting correlations across CNN semantic levels. ICPR 2020 workshop; ~2,000 citations.
- **PatchCore** (Roth et al., CVPR 2022) — a "maximally representative memory bank" of nominal patch features + kNN scoring. Hit ~99.6% image-level AUROC on the original MVTec AD, halving the prior error. Open-source (`amazon-research/patchcore-inspection`; also implemented in Intel's **Anomalib**).

**Known weakness:** inference cost scales with the memory bank size (kNN), and ImageNet-pretrained backbones are biased toward natural images, causing false detections on industrial imagery (known domain-mismatch failure mode).

## 3. Student–teacher / distillation methods

Train a *student* to mimic a frozen pretrained *teacher* on normal images; anomalies appear where the student fails.

- **Knowledge-distillation AD** (Bergmann et al., CVPR 2020) — original student–teacher framing.
- **Reverse Distillation** (Deng & Li, 2022), **AST** (asymmetric student–teacher).
- **EfficientAD** (Batzner et al., 2023–24, MVTec) — distills a student against a pretrained teacher ("knowledge distillation beyond the normal images") combined with a lightweight global autoencoder; reaches millisecond-level latencies and is the current practical state-of-the-art for real-time inspection.

**Related:** normalizing-flow density estimators (FastFlow, CFLOW) model normal-feature densities directly.

## Benchmark landscape and why MVTec AD 2 exists

- **MVTec AD** (the original, 2019) and **VisA**: saturated. Segmentation AU-PRO exceeds ~97% and top methods compete within ~1 percentage point, which no longer discriminates between models.
- **MVTec AD 2** (Heckler-Kram, Neudeck, Scheler, König, Steger — IJCV 2026 / arXiv:2503.21622): the next-generation benchmark designed to break saturation. Eight scenarios with >8,000 high-res images covering transparent/reflective surfaces, overlapped objects, high normal-variance, extremely small defects, and lighting-condition shifts.

### Published numbers on MVTec AD 2 (context for the write-up)

| Method | Avg AU-PRO(0.05) |
|---|---|
| EfficientAD | ~58.7% |
| PatchCore | ~53.8% |

- Average AU-PRO(0.30) of the seven/eight benchmarked SOTA methods remains **below 60%**; with the stricter AU-PRO(0.05) it drops **below ~30%** on average.
- On hard categories (e.g. Can, Rice), PatchCore-class methods fall **below 30%**.
- A training-free DINOv2-based method (**SuperAD**) won the CVPR 2025 VAND 3.0 challenge on the public splits.

The README's claim that SOTA tops out under ~60% average AU-PRO is accurate, and the dataset genuinely leaves room to demonstrate model-design decisions rather than merely re-benchmarking a saturated dataset.

## Positioning of this project

Existing methods solve the same cold-start problem (fit on normal images only, detect unseen defects), but each family has documented, teachable failure modes. ScanGuard's contribution is the *comparison*: how far a plain reconstruction baseline falls short versus a patch-feature / embedding method on an unsatured benchmark, plus what the failure modes reveal about the underlying assumptions.

## Key references

- Heckler-Kram, Neudeck, Scheler, König, Steger — *The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection*. IJCV 134(4), 2026. arXiv:2503.21622.
- Roth et al. — *Towards Total Recall in Industrial Anomaly Detection* (PatchCore). CVPR 2022. arXiv:2106.08265.
- Defard et al. — *PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization*. arXiv:2011.08785.
- Batzner et al. — *EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies*. arXiv:2303.14535.
- Bergmann et al. — *Uninformed Students: Student–Teacher Anomaly Detection with Discriminative Latent Embeddings*. CVPR 2020.
- Zavrtanik et al. — *DRAEM — A discriminatively trained reconstruction embedding for surface anomaly detection*. ICCV 2021.
- SuperAD technical report, CVPR 2025 VAND 3.0 challenge. arXiv:2505.19750.