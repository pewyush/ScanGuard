# ScanGuard: Unsupervised Visual Anomaly Detection for Industrial Quality Control

**Lab Report Submission — 5 October 2026**

| Field | Detail |
|---|---|
| Project title | ScanGuard — Unsupervised Visual Anomaly Detection for Industrial Quality Control |
| Submission date | 5 October 2026 |
| Team members | Peeyush, Harshit Batra |
| Repository | `ScanGuard` (this project) |
| Reference documents | `README.md`, `docs/review-of-existing-systems.md`, `docs/objectives-and-methodology.md`, `docs/algorithm-technique-relevance.md`, `docs/feasibility.md` |
| Dataset | MVTec AD 2 (MVTec Software GmbH), CC BY-NC-SA 4.0 |
| Environment | Python 3.12.10, single consumer GPU |

---

## 1. Abstract

Industrial quality control depends on catching defects accurately and consistently, yet conventional inspection relies on manual examination or hand-crafted rules — both time-consuming, subjective, and unscalable. Deep learning has improved automated inspection, but the standard supervised formulation (train a classifier on labelled *defect vs. no defect* pairs) is poorly matched to real manufacturing, because normal parts are abundant while each individual defect type is rare, diverse, and often unanticipated at design time.

**ScanGuard** therefore frames automated visual inspection as **unsupervised anomaly detection**: learn the distribution of *normal* appearance from defect-free images only, then flag previously unseen defects by how poorly they fit that distribution. The project centres on one research question:

> Can a deep learning model learn the normal visual characteristics of a manufactured product well enough to detect previously unseen defects, without being trained on labeled defect examples?

To answer it, two method families are built and compared on the **MVTec AD 2** benchmark: a deliberately weak **convolutional autoencoder** baseline scored by reconstruction error, and a stronger **embedding/distribution-based** detector (pretrained CNN patch features with per-position multivariate Gaussians and Mahalanobis scoring, in the spirit of PaDiM). Evaluation covers image-level detection (AUROC, precision/recall, F1) and pixel-level localization (AU-PRO, with pixel-AUROC as a fallback), plus qualitative anomaly-map overlays against ground-truth masks.

MVTec AD 2 was selected because the earlier MVTec AD and VisA benchmarks are saturated — segmentation AU-PRO exceeds ~97% and leading methods compete within roughly one percentage point, which no longer discriminates between models. MVTec AD 2 deliberately stresses known failure modes (transparent and reflective surfaces, bulk/overlapping objects, high variance among normal samples, extremely small defects, lighting shifts between splits), leaving genuine room to demonstrate model-design reasoning rather than replicate published scores. Published state of the art averages under ~60% average AU-PRO on it, so the comparison has room to say something.

The expected and desired outcome is **not a state-of-the-art claim**. It is a controlled, reproducible demonstration of *how far* a purely reconstruction-based approach falls short of a feature-embedding approach on an unsaturated benchmark, and an analysis of *why* — grounded in failure modes the source papers themselves document.

---

## 2. Problem Statement

### 2.1 The industrial context

Quality control is a mandatory stage of manufacturing. Identifying defects accurately and consistently is critical for maintaining product quality and reducing production losses, yet defects vary considerably in appearance, size, shape, and location — scratches, cracks, dents, holes, surface contamination, structural abnormalities. Catching all of them with simple image-processing techniques is difficult by construction, because the rule set would have to enumerate every possible defect in advance.

### 2.2 Why existing inspection approaches fall short

| Existing approach | Limitation |
|---|---|
| Manual human inspection | Slow, costly, subjective, inconsistent between inspectors, hard to scale |
| Rule-based / classical image processing | Requires hand-engineered features per defect type; brittle to lighting and surface variation; cannot anticipate unseen defects |
| Supervised deep learning classifiers | Requires large labelled sets of defective examples; rare defect types make labels expensive or unobtainable; poor generalisation to defect types absent from training |

### 2.3 The formulation gap

The supervised framing conflicts with how manufacturing data actually exists: normal products are abundant, but any individual defect type is rare, and defect types themselves are diverse and hard to enumerate in advance. Training a model to classify *defect vs. no defect* therefore optimises for a label supply that does not exist in practice.

### 2.4 The reframed task

Rather than defect classification, ScanGuard treats inspection as anomaly detection on normal images only:

```
Normal product images (defect-free only)
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

This is the **cold-start problem** of unsupervised visual anomaly detection: fit a system on normal images only, then detect and localize previously unseen defects at test time, with no labelled defect examples ever seen during training.

### 2.5 Scope boundary

This project is scoped around the **AI/ML research question** — model design, training, and evaluation of anomaly detection performance. It explicitly excludes model deployment and serving infrastructure, real-time inference pipelines, MLOps tooling (CI/CD, monitoring, drift detection), and production-grade explainability dashboards. These are natural follow-up extensions but are not part of this project's evaluation criteria.

### 2.6 Benchmark choice and its justification

**MVTec AD 2** (Heckler-Kram et al., IJCV 2026) is the second-generation successor to the original MVTec AD benchmark, purpose-built because top anomaly detection methods had begun saturating the original dataset.

| Property | Value | Why it matters here |
|---|---|---|
| Scenarios | 8 industrial inspection scenarios | Covers objects and materials absent from MVTec AD |
| Images | 8,000+ high-resolution (2.6–5 MP) | Sufficient data for feature-distribution modelling |
| Training split | Defect-free images only, per scenario | Exactly matches the unsupervised setup |
| Validation split | Defect-free images only | Supports threshold selection without defects |
| Test splits | Two: public (pixel-precise GT masks) and private (GT via MVTec benchmark server) | Public split enables local evaluation; private split allows an unbiased final check |
| Difficult conditions | Transparent/reflective surfaces, dark-field and back-light illumination, overlapped/bulk objects, high intra-class variance in normal samples, extremely small defects, lighting-condition shift between splits | These are precisely the settings where documented failure modes appear |
| Headroom | SOTA average AU-PRO below ~60%; below ~30% under the stricter AU-PRO(0.05) | Room to demonstrate design decisions rather than re-benchmark a saturated dataset |
| Licence | CC BY-NC-SA 4.0 (non-commercial, academic) | Permits coursework use |

Accessibility was verified: the dataset is free to download via a short approval form on mvtec.com with no payment, ground truth for the public split ships with the data, and official PyTorch data-loading and submission utilities are published. The only external dependency is MVTec's benchmark server, needed solely for the optional private split.

---

## 3. Proposed Solution and Methodology

### 3.1 Proposed solution in one line

Compare a **reconstruction-based convolutional autoencoder** against a **patch-embedding distribution model (PaDiM-style)** on MVTec AD 2, scoring both at image and pixel level, and analyse the gap between them in terms of each family's underlying assumption about normality.

### 3.2 Why a comparison, not a single model

Every method family reviewed solves the *same* cold-start problem; the families differ in **what they define as "normality"** and **which failure mode they trade away**. A single model would report a score without explaining anything. The comparison isolates the design axis: does *reconstructing normals pixel-by-pixel* suffice, or must normality be modelled in a pretrained feature space?

### 3.3 Methodology — phased plan

#### Phase 1 — Problem framing and literature grounding
- Review the core approaches: reconstruction-based (autoencoders, GANs, DRAEM-style), embedding/feature-based (SPADE, PaDiM, PatchCore), student–teacher/distillation, and density-based (normalizing flows).
- For each reviewed line of work, state its research **objective**, its **methodology**, and the **core assumption** it rests on (see `docs/objectives-and-methodology.md`).
- **Select 1–3 scenario categories** from MVTec AD 2, chosen to *contrast* a "reasonable" scenario with a "hard" one (transparent/reflective, or small-defect), so both failure modes have something to bind to. Running all 8 categories is a time/compute risk and is not committed to.

#### Phase 2 — Data pipeline and exploratory analysis
- Load the chosen categories; inspect normal versus defective samples, native resolution, lighting variation, and defect size distribution.
- Build the preprocessing pipeline: resizing to the input resolution required by the chosen backbone, per-channel normalisation, and augmentation applied **only to the normal-only training set**.
- Record dataset statistics and example anomaly distributions to support the qualitative analysis.

#### Phase 3 — Baseline model (reconstruction)
- Train a **convolutional autoencoder** (encoder → bottleneck → decoder) on normal images only, with an MSE reconstruction loss.
- Derive the anomaly score from reconstruction error: per-pixel |input − reconstruction|, optionally weighted by structural similarity, aggregated into an anomaly map and then into an image-level score.
- Establish baseline image-level metrics (AUROC, precision/recall, F1 for normal vs. anomalous).
- **Expectation, stated in advance:** this family is documented to be the weakest modern approach, because autoencoders *over-generalize* and reconstruct defective regions too well. Demonstrating this is the project's core negative-result narrative, not a failure to be hidden.

#### Phase 4 — Stronger comparison model (embedding)
- Extract multilevel patch features from a frozen pretrained CNN backbone.
- Fit a **per-position multivariate Gaussian** (mean vector plus regularised covariance guaranteeing full rank/invertibility) to the normal training embeddings — a purely statistical fit, no backprop, contrasting directly with the autoencoder's full training loop.
- At inference, score each test patch by its **Mahalanobis distance** to that position's Gaussian; the per-position distances form the anomaly map, and the maximum gives the image-level score.
- Where budget allows, extend to a PatchCore-style memory bank with greedy coreset subsampling and kNN patch scoring, reported as a literature anchor and, if implemented, as a third comparison point.

#### Phase 5 — Localization study
- Determine whether pixel-level anomaly maps come from the same model at no extra cost (the embedding model) or require separate machinery (the reconstruction baseline).
- This is the concrete axis on which the two compared methods differ, and a core finding of the write-up.

#### Phase 6 — Evaluation
- **Image-level:** AUROC, precision-recall, F1.
- **Pixel-level:** **AU-PRO** as the primary localization metric, computed with MVTec's official utilities; **pixel-AUROC** as a fallback only if those utilities prove unwieldy. Report AU-PRO(0.05) as well as AU-PRO(0.30) for honest comparison with published numbers.
  - *Why AU-PRO:* plain pixel AUROC is inflated by extreme class imbalance, since anomalies are tiny. Per-Region Overlap computes region-scoped recall per connected ground-truth region, averaged across regions, so every defect counts equally regardless of size.
- **Qualitative:** overlay anomaly heatmaps against ground-truth masks for a sample of defective images.
- **Cross-category comparison:** discuss where each approach succeeds and where it struggles (small defects, transparent surfaces, lighting shift).

#### Phase 7 — Analysis and write-up
- Discuss failure modes and what they reveal about the limits of reconstruction-based anomaly detection.
- Compare results against published MVTec AD 2 benchmark numbers.
- Frame the autoencoder as a deliberate weak baseline and a floor, not a target for optimisation.
- Explicitly scope deployment concerns (serving, latency, monitoring) as future work.

### 3.4 Methodology summary table

| Phase | Activity | Primary output |
|---|---|---|
| 1 | Literature review; category selection | Grounded method taxonomy; chosen 1–3 categories |
| 2 | Data loading, EDA, preprocessing | Reproducible preprocessing pipeline; dataset observations |
| 3 | Convolutional autoencoder on normals only | Baseline anomaly scores and image-level metrics |
| 4 | Patch-feature Gaussians + Mahalanobis scoring | Comparison-model anomaly maps and metrics |
| 5 | Localization study | Finding on whether localization is free or separate |
| 6 | AUROC / PR / F1 / AU-PRO / qualitative | Results tables; heatmap-vs-mask visualisations |
| 7 | Failure-mode analysis | Discussion of documented assumptions; future work |

### 3.5 Compute and tooling

All three candidate method families run on a single consumer GPU:

| Approach | Training required | GPU footprint |
|---|---|---|
| Convolutional autoencoder (baseline) | Full training, small model | Low |
| Patch-embedding / PaDiM-style | None (feature extraction + statistical fit) | Low–moderate; inference cost independent of training-set size |
| PatchCore-style memory bank | None | Low–moderate; kNN inference cost scales with bank size |
| EfficientAD-style student–teacher (reference only) | Modest training | Moderate |

Current environment: Python 3.12.10 with NumPy, pandas, scikit-learn, Matplotlib, seaborn, and Jupyter/Notebook. Deep-learning and dataset-specific dependencies (PyTorch, MVTec utilities) are added when the implementation phase begins.

### 3.6 Risk register and fallbacks

| # | Risk | Likelihood / impact | Mitigation and fallback |
|---|---|---|---|
| 1 | Reconstruction baseline performs poorly — autoencoders reconstruct defects too well, and on small/transparent/high-variance defects numbers may be near-unusable | High likelihood / high impact (expected) | Position explicitly as a methodology demonstration, not a failure; do not over-interpret the baseline. Fallback: switch baseline to a VAE or small memory-based method while keeping the discussion of why image-level AE detection fails |
| 2 | Chosen categories yield little signal — hardest scenarios push even PatchCore below ~30% AU-PRO | Medium / high | Prefer a mix of one "reasonable" and one "hard" category rather than all-hard; limit to 1–3 categories |
| 3 | ImageNet-feature domain bias — pretrained backbones are misaligned with industrial imagery, a known false-detection failure mode | High / medium | Expected; report as a limitation and future work rather than a blocker |
| 4 | External dependencies — dataset download gated by an approval form; benchmark-server availability for the private split | Medium / medium | Kick off the download form early; treat private-split evaluation as optional |
| 5 | GPU unavailability or time exhaustion | Low–medium / high | Fallback: smaller autoencoder + one embedding method on fewer categories; pre-extract features (cheap); compare image-level AUROC only |
| 6 | Localization metrics prove unwieldy | Medium / low | Fall back to pixel-AUROC; use official utilities for AU-PRO as planned |
| 7 | Scope creep toward deployment/MLOps | Medium / medium | Already excluded in project scope; not an evaluation criterion |

---

## 4. Objectives

### 4.1 Primary objective

To determine, through a controlled comparison on an unsaturated industrial benchmark, **how far a purely reconstruction-based deep learning model can detect and localize previously unseen defects from normal-only training, relative to a feature-embedding approach** — and to explain the gap in terms of the assumptions each family makes about normality.

### 4.2 Specific objectives

| # | Objective | Success indicator |
|---|---|---|
| O1 | Implement and train a convolutional autoencoder baseline on normal-only images from MVTec AD 2 | Trained model producing per-pixel reconstruction residuals on chosen categories |
| O2 | Derive and calibrate anomaly scores that convert reconstruction error into both an image-level defect flag and a pixel-level anomaly map | Documented scoring function; thresholds selected on the defect-free validation split |
| O3 | Implement a stronger embedding-based detector (pretrained patch features + per-position multivariate Gaussians + Mahalanobis scoring) | Anomaly maps and image-level scores produced without per-dataset gradient training |
| O4 | Quantify the performance gap between the two approaches at image level | AUROC, precision-recall and F1 for both methods on all chosen categories |
| O5 | Quantify the localization gap at pixel level | AU-PRO(0.30) and AU-PRO(0.05), with pixel-AUROC as fallback; qualitative heatmap-vs-mask overlays |
| O6 | Characterise failure modes and tie them to the assumptions of each family | Documented failure analysis linking observed behaviour to literature-documented causes (AE over-generalization; ImageNet domain bias) |
| O7 | Position results against published MVTec AD 2 benchmark numbers | Comparison table vs. published EfficientAD / PatchCore results |
| O8 | Maintain reproducibility and honest scope | Seeded runs, fixed data pipeline, documented fallbacks; deployment concerns explicitly excluded |

### 4.3 Explicit non-objectives

- Achieving state-of-the-art results on MVTec AD 2.
- Comparing many method families — the reconstruction-versus-embedding contrast is *the* required comparison; other techniques are benchmark anchors, discussion points, or out-of-scope extensions.
- Any production concern: serving, latency budgets, real-time pipelines, MLOps tooling, drift monitoring, explainability dashboards.

### 4.4 Contribution statement

ScanGuard's contribution is **the comparison itself**: how far a plain reconstruction baseline falls short of a patch-feature embedding method on an unsaturated benchmark, plus what the observed failure modes reveal about the underlying assumptions of each family. Every critique used is one the source papers themselves state, so the project's demonstration rests on established literature rather than an invented objection.

---

## 5. Relevant Algorithms and Techniques to Be Used

The relevance test applied to every algorithm below: does it (a) illuminate the research question as a **compared method**, (b) provide a **baseline/floor**, or (c) provide **benchmark context** against which both are judged?

### 5.1 Algorithms adopted for implementation

#### 5.1.1 Convolutional autoencoder with reconstruction-error scoring — *baseline / floor*

- **Objective:** learn to regenerate normal appearance; expect anomalous input to reconstruct poorly, making reconstruction error a natural anomaly score.
- **Technique:** train encoder → bottleneck → decoder on normal images only with an MSE reconstruction loss. At test time the per-pixel difference (input − reconstruction), optionally weighted by structural similarity, becomes the anomaly map; its aggregation gives the image-level score.
- **Core assumption:** the autoencoder's capacity is too small to reproduce defective content it never saw.
- **Documented limitation:** autoencoders *over-generalize* — they reconstruct defective regions too well, so the residual is small exactly where the defect is. This is the baseline's expected weakness and the reason image-level detection with a plain autoencoder is the floor rather than a target.
- **Role in this project:** compared baseline (floor).

#### 5.1.2 Deep feature reconstruction (DFR) — *optional variant of the baseline*

- **Objective:** keep the reconstruction framing but score at the *feature* level instead of the pixel level, reducing sensitivity to texture and colour aliasing.
- **Technique:** the autoencoder reconstructs the feature activations of a frozen pretrained CNN rather than pixels; the L2 distance between predicted and extracted features is the anomaly score.
- **Role:** optional extension if pixel-space residuals prove dominated by aliasing artefacts.

#### 5.1.3 PaDiM-style patch distribution modelling — *stronger comparison method*

- **Objective:** encode the normal patch-feature *distribution* rather than just exemplars, exploiting correlations across CNN semantic levels for better localization, at minimal training cost.
- **Technique:**
  1. Extract embeddings at several pretrained-CNN layers (e.g. ResNet layers 1–3), with random dimensionality reduction per position.
  2. For each spatial patch position, fit a **multivariate Gaussian** to the training embeddings — mean vector plus covariance matrix with a regularising term guaranteeing full rank and invertibility.
  3. At inference, compute the **Mahalanobis distance** of each test patch to its position's Gaussian; per-position distances form the anomaly map; the maximum gives the image-level score.
- **Core assumption:** per-position normal patch features are approximately Gaussian, and the cross-level covariance captures the semantic structure of normals.
- **Why it is the right "stronger baseline":** training is purely statistical with no backprop, keeping the experiment on budget while directly contrasting with the autoencoder's full training loop; localization is inherent rather than bolted on; and on MVTec AD 2 its published performance still falls below ~30% on hard categories, making it a genuinely informative comparison rather than a trivially winning SOTA.
- **Role:** compared method (stronger baseline).

#### 5.1.4 PatchCore-style memory bank — *third point of comparison, budget permitting*

- **Objective:** cold-start detection and localization with "total recall" of normal appearance while keeping inference practical (99.6% image-level AUROC on the original MVTec AD, halving the next-best error).
- **Technique:**
  1. Extract **locally aware patch features** from a frozen pretrained backbone (e.g. WideResNet-50), aggregating multiple feature hierarchies with a random linear projection into patch descriptors.
  2. Store all normal patch features in a **memory bank** — a "hyper-dimensional space of normality".
  3. Apply **greedy coreset subsampling** to approximate the bank with a small, maximally representative subset, cutting storage and inference cost while keeping or slightly improving accuracy.
  4. At test time score each test patch by distance to its nearest neighbour in the bank; build the anomaly map from the top-scoring patch and its neighbours; image score from the maximum patch score.
- **Core assumption:** a representative set of normal patch features suffices — anomalies are, by construction, far from every stored normal patch.
- **Documented weaknesses:** inference cost scales with bank size (kNN), and ImageNet-pretrained features are biased toward natural imagery, producing false positives on industrial, transparent and reflective surfaces. Both matter on MVTec AD 2.
- **Role:** optional third comparison point; primarily a literature anchor.

#### 5.1.5 Evaluation metrics — *measurement layer*

| Level | Metric | Role |
|---|---|---|
| Image | AUROC, precision-recall, F1 | Primary detection result for both compared methods |
| Pixel | **AU-PRO(0.30)** | Primary localization metric — region-scoped and size-fair, and the metric all published MVTec AD 2 numbers use |
| Pixel | **AU-PRO(0.05)** | Honest ceiling; integrates the PRO curve over FPR ∈ [0, 0.05] so only very precise maps score well |
| Pixel | Pixel-AUROC | Fallback only, if official AU-PRO utilities prove unwieldy — inflated by class imbalance since defects are tiny |
| Qualitative | Anomaly heatmap vs. ground-truth mask overlays | Failure-mode evidence |

### 5.2 Algorithms retained as benchmark anchors or discussion points (not implemented)

| Technique | Why relevant | Direction of relevance |
|---|---|---|
| **DRAEM** (discriminative reconstruction with synthetic anomalies) | Directly addresses the autoencoder's documented over-generalization problem by making detection discriminative; its motivating observation is the very failure this project expects to reproduce | Discussion / future work. Out of scope for implementation because anomaly synthesis would convert the unsupervised setup into a synthetic-discriminative one and confound the "seen no defects" narrative |
| **Knowledge distillation / Reverse Distillation / AST / EfficientAD** (student–teacher) | EfficientAD is the practical state-of-the-art for real-time inspection and the best published average on MVTec AD 2 (58.7% AU-PRO(0.30); 30.8% AU-PRO(0.05)); its design choices map one-to-one onto the documented failure modes of the other families | Benchmark anchor via reported numbers, plus future work. Too much training and engineering (distillation, anti-mimicry loss, calibration) to reproduce within budget |
| **Normalizing flows** (FastFlow, CFLOW) | Alternative formulation of normality as explicit feature density | Discussion only. Density estimation risks mis-scoring legitimately diverse normal samples as anomalous — exactly MVTec AD 2's high-variance regime |
| **SuperAD / DINOv2 training-free features** | Strong recent result (CVPR 2025 VAND 3.0 challenge winner on public splits); evidence that pretrained feature quality is decisive | Optional cheap third comparison if GPU allows; heavy ViT-L backbone (~300M params) otherwise |
| **GAN-based reconstruction** (AnoGAN, f-AnoGAN) | Historical generative alternative to the autoencoder | Mentioned in review, not implemented — unstable training, same ancestor failure mode, no added teaching value |
| **MVTec AD 2 split and metric design** | Its design choices — high normal variance, small defects, transparent/reflective objects, lighting shift — are exactly the settings where autoencoder over-generalization and ImageNet domain bias surface | Evaluation context; drives category selection |

### 5.3 Cross-family summary of the assumption each technique rests on

| Family | Objective it optimises for | The single assumption everything rests on |
|---|---|---|
| Reconstruction (AE / VAE / GAN) | Faithful normal reconstruction | Anomalies are unreconstructable with limited capacity — *known to be false* for small and high-variance defects |
| DRAEM | Discriminative margin via synthetic defects | A boundary learned on pseudo-defects generalises to real defects |
| Embedding (PaDiM / PatchCore) | Distance to the normal-feature distribution | Frozen pretrained features separate normal from defective patches |
| Student–teacher (EfficientAD) | Cheap asymmetry between networks | A student forced to mimic normals only cannot mimic anomalies |
| Density (FastFlow / CFLOW) | Log-density of normal features | Normal features are well modelled by a flow to a simple base distribution |

### 5.4 Concrete decisions implied by the analysis

1. **Baseline = plain convolutional autoencoder** on normal-only training, scored by pixel-wise (and optionally feature-wise) reconstruction error.
2. **Stronger comparison = a PaDiM-style method** (pretrained CNN embedding + per-position multivariate Gaussians + Mahalanobis distance), with PatchCore reported as a literature anchor unless extra budget appears.
3. **Categories: a deliberate mix** — one "reasonable" and one transparent/reflective or small-defect scenario. All-hard categories would push even strong methods below ~30% and leave the comparison with nothing to discuss.
4. **Metrics:** image-level AUROC / PR / F1 as primary; AU-PRO via official utilities as the localization primary; pixel-AUROC as fallback; report AU-PRO(0.05) alongside AU-PRO(0.30).
5. **Narrative grounded in the literature's own admissions** — autoencoder over-generalization, ImageNet domain bias, and benchmark saturation are all stated in the reviewed papers, so the demonstration is empirical rather than invented.

---

## 6. References

1. Heckler-Kram, M., Neudeck, M., Scheler, C., König, R., & Steger, G. *The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection.* International Journal of Computer Vision (IJCV), 134(4), 2026. arXiv:2503.21622. Dataset: https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2
2. Bergmann, P., Fauser, M., Sattlegger, S., & Steger, G. *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.* IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.
3. Bergmann, P., Batzner, K., Fauser, M., Sattlegger, S., & Steger, G. *Beyond Dents and Scratches: Logical Constraints in Unsupervised Anomaly Detection and Localization.* IJCV, 2022. *(Introduces the AU-PRO metric.)*
4. Roth, K., Pemula, T., Zepeda, J., Schölkopf, B., Brox, T., & Gehler, X. *Towards Total Recall in Industrial Anomaly Detection* (PatchCore). CVPR, 2022. arXiv:2106.08265. Code: `amazon-research/patchcore-inspection`
5. Defard, T., Setkov, V., Loesch, K., & Audigier, D. *PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection and Localization.* IEEE International Conference on Pattern Recognition Workshops (ICPR-W), 2021. arXiv:2011.08785.
6. Batzner, K., Heckler, H., & König, R. *EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies.* IEEE/CVF Winter Conference on Computer Vision (WACV), 2024. arXiv:2303.14535.
7. Bergmann, P., et al. *Uninformed Students: Student–Teacher Anomaly Detection with Discriminative Latent Embeddings.* CVPR, 2020.
8. Zavrtanik, V., Kristan, M., & Skočaj, D. *DRAEM: A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection.* IEEE/CVF International Conference on Computer Vision (ICCV), 2021. arXiv:2108.07610.
9. Deng, H. & Li, X. *Anomaly Detection via Reverse Distillation from One-Class Embedding.* IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 9737–9746.
10. Zhang, H., et al. *SuperAD: A Training-free Anomaly Classification and Segmentation Method.* CVPR 2025 VAND 3.0 Challenge report. arXiv:2505.19750.
11. Oquab, M., et al. *DINOv2: Learning Robust Visual Features without Supervision.* arXiv:2304.07193.
12. Schlegl, T., Seeböck, P., Waldstein, S. M., Schmidt-Erfurth, U., & Langs, G. *Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery* (**AnoGAN**). Information Processing in Medical Imaging (IPMI), Springer LNCS, 2017. arXiv:1703.05921. **Note:** AnoGAN is frequently miscited as "ICANN 2018" and its first author is frequently misrendered as "Scherr"; both are incorrect.
13. Schlegl, T., Seeböck, P., Waldstein, S. M., Schmidt-Erfurth, U., & Langs, G. *f-AnoGAN: Fast Unsupervised Anomaly Detection with Generative Adversarial Networks.* Medical Image Analysis, 54, pp. 30–44, 2019. **Note:** frequently miscited as "BMVC 2019"; the venue is Medical Image Analysis.
14. Yu, J., Zheng, Y., Wang, X., Li, W., Wu, Y., Zhao, R., & Wu, L. *FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows.* arXiv:2111.07677, 2021.
15. Gudovskiy, D., Ishizaka, S., & Kozuka, K. *CFLOW-AD: Real-Time Unsupervised Anomaly Detection with Localization via Conditional Normalizing Flows.* arXiv:2107.12571, 2021.
16. Zhou, Y., Xu, X., Song, J., Shen, F., & Shen, H. T. *MSFlow: Multiscale Flow-Based Framework for Unsupervised Anomaly Detection.* IEEE Transactions on Neural Networks and Learning Systems, 2024. *(MSFlow is the flow-based method actually benchmarked on MVTec AD 2.)*
17. Zavrtanik, V., Kristan, M., & Skočaj, D. *DSR — A Dual Subspace Re-Projection Network for Surface Anomaly Detection.* ECCV, 2022, pp. 539–554. *(The DSR baseline benchmarked on MVTec AD 2, by the DRAEM authors.)*
18. Liu, Z., Zhou, Y., Xu, Y., & Wang, Z. *SimpleNet: A Simple Network for Image Anomaly Detection and Localization.* CVPR, 2023, pp. 20402–20411.
19. Tien, T. D., Nguyen, A. T., Tran, N. H., Huy, T. D., Duong, S. T., Nguyen, C. D., & Truong, H. G. *Revisiting Reverse Distillation for Anomaly Detection* (**RD++**). CVPR, 2023, pp. 24511–24520.
20. Kingma, D. P. & Welling, M. *Auto-Encoding Variational Bayes.* ICLR, 2014. *(VAE.)*
21. Akcay, S., Ameln, D., Vaidya, A., Lakshmanan, B., Ahuja, N., & Genc, U. *Anomalib: A Deep Learning Library for Anomaly Detection.* IEEE Symposium on Series on Data, Engineering and Applications (SSDEA), 2022. arXiv:2202.08341. *(Reference implementation library containing PaDiM, PatchCore, FastFlow and DSR.)*
22. MVTec Software GmbH. *MVTec AD 2 Benchmark Server.* https://www.mvtec.com/benchmark *(private-split ground-truth evaluation; optional for this project).*

### 6.1 Internal project documents

- `README.md` — project scope, dataset rationale, tentative plan, out-of-scope list, system requirements.
- `docs/review-of-existing-systems.md` — review of the three dominant method families and the benchmark landscape.
- `docs/objectives-and-methodology.md` — objective, methodology, and core assumption per reviewed line of work.
- `docs/algorithm-technique-relevance.md` — mapping of algorithms onto the project plan, with relevance direction per item.
- `docs/feasibility.md` — data accessibility, compute budget, risk timeline, and fallback plans.

### 6.2 Published benchmark figures (verified against arXiv:2503.21622)

The MVTec AD 2 paper benchmarks **seven** methods: EfficientAD, Reverse Distillation (RD), Reverse Distillation Revisited (RD++), PatchCore, MSFlow, SimpleNet and Dual Subspace Re-Projection (DSR). All results below are at input resolution 256 × 256, on the private test split (TEST priv).

**Average anomaly segmentation AU-PRO(0.30)** — the figures quoted in this report:

| Method | Avg. AU-PRO(0.30) |
|---|---|
| EfficientAD | **58.7%** |
| RD++ (Revisiting Reverse Distillation) | 54.3% |
| **PatchCore** | **53.8%** |
| RD (Reverse Distillation) | 53.0% |
| MSFlow | 52.7% |
| DSR | 49.0% |
| SimpleNet | 46.4% |
| **Mean of all seven** | **52.6%** |

**Average AU-PRO(0.05)** — the stricter metric: EfficientAD is best at **30.8%**, and **every** benchmarked method stays **below 31%**.

Three corrections of note, applied throughout this report:

1. **The ≈58.7% / ≈53.8% figures are AU-PRO(0.30), not AU-PRO(0.05).** Two internal project documents disagreed on this; the paper's Table IX settles it.
2. **The "below ~30% on hard categories" claim is an AU-PRO(0.05) statement** and should be quoted as such. At AU-PRO(0.05) on TEST priv, PatchCore scores 4.7% on *Can* and 25.6% on *Rice*; at AU-PRO(0.30) the same cells read 21.6% and 50.9%. *Can* is the hardest scenario for every method investigated, not only for memory-bank methods.
3. **The 99.6% image-level AUROC figure belongs to the original MVTec AD**, not to MVTec AD 2, and must never be presented as an MVTec AD 2 result.

Two further caveats worth carrying into the write-up:

- **Input resolution dominates the numbers.** The paper's appendix shows PatchCore rising from 28.8% to 62.3% AU-PRO(0.05) on *Rice* when resolution is increased, at the cost of roughly an order of magnitude in runtime and memory. Any ScanGuard result must therefore state its input resolution, or it is not comparable.
- **Threshold-independent metrics alone are insufficient.** The paper reports that PatchCore attains strong AU-PRO yet the worst image-level F1 (21.8% for the best method), because anomaly maps are by default normalised across the whole dataset to [0, 1] — a procedure that breaks when normal and anomalous images are imbalanced. Threshold-dependent metrics should be reported alongside AU-PRO.

Finally, the saturation claim is accurate as stated in the project docs: on MVTec AD (5,354 images, 15 categories) the best model reaches **97.8%** mean AU-PRO, and leading methods compete within roughly one percentage point. VisA contains 10,821 images across 12 categories.

---

## Appendix A — Expected Deliverables

| Deliverable | Description |
|---|---|
| Data pipeline | Reproducible loading and preprocessing for chosen MVTec AD 2 categories |
| Baseline model | Trained convolutional autoencoder + documented anomaly-scoring function |
| Comparison model | Patch-feature Gaussian model with Mahalanobis scoring |
| Results tables | Image-level and pixel-level metrics for both methods, all chosen categories |
| Qualitative results | Anomaly heatmaps overlaid on ground-truth masks |
| Analysis | Failure-mode discussion tied to each family's documented assumption |
| Comparison to SOTA | Positioning against published MVTec AD 2 numbers |

## Appendix B — Risk Summary (one-line view)

The project is **feasible within scope** provided (a) the experiment is limited to 1–3 scenario categories, (b) the autoencoder is positioned as a deliberate weak baseline rather than an optimisation target, and (c) external dependencies (dataset access form, optional benchmark server) are initiated early. The honest deliverable is a methodology comparison on an unsaturated benchmark — **not a state-of-the-art claim.**