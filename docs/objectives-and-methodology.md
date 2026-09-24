# Objectives and Methodology of Current Work

*Companion artifact to `review-of-existing-systems.md`. States, for each reviewed line of work, the research **objective** (what problem it set out to solve), the **methodology** (how it achieves that objective), and the core assumption the method rests on. It answers "why does each approach exist, and how does it actually work?"*

The shared objective across all of this work is the **cold-start problem of unsupervised visual anomaly detection**: fit a system on *normal* (defect-free) images only, then detect and localize previously unseen defects at test time, with no labeled defect examples ever seen in training.

---

## 1. Reconstruction-based methods

### Autoencoders / VAEs (pixel-space baseline)
- **Objective:** Learn to regenerate normal appearance; expect anomalous input to be reconstructed poorly, making reconstruction error a natural anomaly score.
- **Methodology:**
  1. Train encoder→bottleneck→decoder on an MSE (or similar) reconstruction loss using normal images only.
  2. At test time, the per-pixel difference (input − reconstruction), optionally with structural similarity weighting, becomes the anomaly map; its aggregation gives the image-level score.
- **Core assumption:** the model capacity of the AE is too small to reproduce defective content it never saw.
- **Documented failure:** AEs often *over-generalize* — they reconstruct defective regions too well, so the residual is small exactly where the defect is (DRAEM, Zavrtanik et al. 2021, cites "autoencoders over-generalize to anomalies" as a core motivation). This is why image-level detection with a plain AE is the weak baseline.

### GAN-based (AnoGAN, f-AnoGAN)
- **Objective:** replace the pixelwise AE with a generative model that maps normals to a latent space; flag inputs that do not lie on the learned normal latent manifold.
- **Methodology:**
  1. Train a GAN (typically WGAN) on normal images so the generator produces them and the discriminator judges them.
  2. At inference, project a test image into the latent space via iterative optimization (AnoGAN) or an encoder (f-AnoGAN), then score by a combination of reconstruction error and a "discriminative" residual from the discriminator.
- **Core assumption:** anomalies fall off the normal data manifold and cannot be mapped back onto it cleanly.
- **Practical note:** heavier and less stable to train than an AE; largely superseded in industrial practice for the same ancestor failure modes.

### DRAEM (Zavrtanik, Kristan & Skočaj, ICCV 2021)
- **Objective:** fix the AE over-generalization problem by making detection **discriminative** rather than purely reconstructive, while keeping the anomaly-free-only training setup.
- **Methodology:**
  1. **Anomaly synthesis:** auto-generate "pseudo-defects" on normal images (Perlin noise, glass color/texture blotches, etc.) — deliberately *just-out-of-distribution* appearances, not faithful reproductions of real defects.
  2. **Reconstructive sub-network:** an AE trained to map the corrupted image back to its clean normal version (implicitly removing the synthetic defect).
  3. **Discriminative sub-network:** a small segmentation head predicts a per-pixel anomaly map from the joint features of the original and reconstructed image, trained with a Dice/Focal loss against the synthetic defect mask.
  4. At test time the network runs in one forward pass; the output is the anomaly map directly — no hand-crafted post-processing. The synthetic masks also yield a fully supervised pixel-level training signal.
- **Core assumption:** a decision boundary learned *jointly* over (input, reconstruction) generalizes better to real defects than either pure reconstruction error or a model over-fit to synthetic appearances alone.
- **Relevance for us:** it is the strongest reconstruction-family method and shows the AE family can be rescued — but it requires an anomaly-synthesis pipeline and discards the "pure" unsupervised assumption the project plan's baseline deliberately keeps.

### Deep feature reconstruction (DFR)
- **Objective:** keep the reconstruction framing but score at the *feature* level instead of pixel level, reducing the sensitivity of pixel-space AEs to texture/color aliasing.
- **Methodology:** an AE reconstructs the feature activations of a frozen pretrained CNN (rather than pixels); the L2 distance between predicted and extracted features is the anomaly score.
- **Core assumption:** reconstructed semantic features are more discriminative for defects than raw pixels while inheriting pretrained representation quality.

---

## 2. Embedding / representation-based methods

Family thesis: *freeze* a strong pretrained network, characterize the distribution of its patch features over normal data, and score test patches by distance to that normal distribution. No per-dataset gradient training.

### SPADE
- **Objective:** pixel-precise localization via nearest-neighbor comparison of pretrained deep features.
- **Methodology:** extract multilevel CNN features (different semantic scales), pool to a common spatial resolution, and for each test patch take the distance to its nearest neighbor among stored normal patches; the resulting per-patch distance map is the anomaly map. Image score = max/aggregate of patch scores.
- **Core assumption:** normal patches cluster densely in feature space; defective patches land in low-density regions with large nearest-neighbor distances.

### PaDiM (Defard et al., ICPR-W 2021)
- **Objective:** encode the normal patch-feature *distribution* (not just exemplars) and exploit correlations **across CNN semantic levels** for better localization, while keeping training cost minimal.
- **Methodology:**
  1. Extract embeddings at several pretrained-CNN layers (e.g. ResNet layers 1, 2, 3 for texture/object categories), with random dimensionality reduction per position.
  2. For each spatial patch position, fit a **multivariate Gaussian** to the training embeddings (mean vector + covariance matrix with a regularizing term to guarantee full rank/invertibility).
  3. At inference, compute the **Mahalanobis distance** of each test patch to its position's Gaussian; that per-position distance forms the anomaly map; the max is used for image-level scoring.
- **Core assumption:** per-position normal patch features are approximately Gaussian; the cross-level covariance captures the semantic structure of normals.
- **Practical note:** train phase is purely statistical (no backprop), inference cost is independent of the training-set size (unlike kNN methods).

### PatchCore (Roth et al., CVPR 2022)
- **Objective:** state-of-the-art cold-start detection/localization with "total recall" of normal appearance, while keeping inference practical. On MVTec AD it reached 99.6% image-level AUROC, halving the error of the next best competitor.
- **Methodology:**
  1. Extract **locally aware patch features** from a frozen pretrained backbone (e.g. WideResNet-50), aggregating multiple feature hierarchies with a random linear projection into patch descriptors.
  2. Store all normal patch features in a **memory bank** ("hyper-dimensional space of normality").
  3. **Greedy coreset subsampling** approximates the memory bank with a small, maximally representative subset — cutting storage and inference cost while keeping (even slightly improving) accuracy.
  4. At test time, each test patch is scored by the distance to its nearest neighbor in the bank; anomaly map from the top-scoring patch and its neighbors; image score from the max patch score.
- **Core assumption:** a representative set of normal patch features is all you need — anomalies are, by construction, far from every stored normal patch.
- **Documented weaknesses:** inference cost scales with bank size (kNN), and ImageNet-pretrained features are biased toward natural imagery, producing false positives on industrial/transparent/reflective surfaces (domain mismatch). Both matter on MVTec AD 2.

---

## 3. Student–teacher / distillation methods

Family thesis: teach a compact *student* to imitate a frozen pretrained *teacher* on normal data; anomalies are regions where the student fails to mimic the teacher.

### Knowledge-distillation AD (Bergmann et al., CVPR 2020)
- **Objective:** detect anomalies from the *differing response* of two networks to normal data.
- **Methodology:** freeze a pretrained teacher CNN; train students (one per teacher layer with different receptive fields) to regress the teacher's feature maps and its "pointwise descriptors" on normal images; at test time, high student–teacher residual ⇒ anomaly.
- **Core assumption:** the student (trained on normals only) cannot generalize its imitation to out-of-distribution content.

### Reverse Distillation (Deng & Li, ECCV 2022) / AST
- **Objective:** avoid teacher collapse and improve separation by inverting the flow — teach a student the *teacher's* features one-way.
- **Methodology:** a one-class bottleneck between teacher and student lets normal features pass through while anomalies break the reconstruction, typically combined with asymmetric/compact student architectures.
- **Core assumption:** an invertible/asymmetric bottleneck is a stricter filter for normality than naive forward imitation.

### EfficientAD (Batzner, Heckler & König, WACV 2024)
- **Objective:** millisecond-level industrial AD (accurate **and** economical for real-time inspection). On MVTec LOCO/MVTec AD it is the current practical state-of-the-art; on MVTec AD 2 it reaches the best published average of the benchmarked SOTA (~58.7% AU-PRO).
- **Methodology:**
  1. **Patch Description Network (PDN):** a lightweight 4-layer CNN whose features for a pixel depend only on a 33×33 patch — features computed in <1 ms on a modern GPU, trained by distilling the WideResNet-101 features PatchCore uses.
  2. **Student–teacher pair:** teacher = frozen distilled PDN; student = PDN trained on normal images. A dedicated training loss ("knowledge distillation beyond the normal images") prevents the student from imitating the teacher *outside* the normal manifold — actually *limiting* how well it can imitate on anomalies, strengthening the residual signal.
  3. **Global autoencoder:** detects *logical* anomalies — invalid combinations/orderings of otherwise-normal local features, which the local student–teacher cannot see.
  4. Calibration step combines the autoencoder output and student–teacher residuals into a final anomaly map (2 ms latency, ~600 images/s on an RTX 2080 Ti-class GPU).
- **Core assumption:** a locally-scoped, distilled student can mirror the normal distribution cheaply but is *deliberately* kept unable to mimic anomalies.
- **Relevance:** demonstrates the engineering axis (fast + accurate) that pure embedding methods trade off; its design choices (PDN distillation, anti-mimicry loss, logical-anomaly AE) each map to a specific documented failure mode of the other two families.

### Normalizing-flow density (FastFlow, CFLOW)
- **Objective:** replace distance-to-distribution with an explicit *density* of normal features.
- **Methodology:** a normalizing-flow model maps normal patch/feature vectors to a simple base distribution; test-time log-likelihood (i.e. negative density) is the anomaly score.
- **Core assumption:** density estimation captures normality sharper than distance metrics; caveat — under high normal variability (MVTec AD 2's high-variance scenarios) density models can overestimate anomality on legitimately diverse normals.

---

## 4. The benchmark: MVTec AD 2 (Heckler-Kram et al., IJCV 2026)

- **Objective:** break the **saturation** of MVTec AD / VisA, where segmentation AU-PRO exceeds ~97% and top methods compete within ~1 point — numbers that no longer discriminate meaningfully between models. The authors aim for a benchmark that *rewards methodological progress*.
- **Methodology (dataset design):**
  1. **Eight scenarios** (8,004 high-res, 2.6–5 MP images) chosen to stress the known failure modes: transparent and reflective surfaces, dark-field/back-light illumination, overlapped/bulk objects, **high variance in normal data**, extremely small defects, and lighting-condition shifts between splits.
  2. **Unsupervised split structure:** train + validation contain *only* non-anomalous images; two test splits (public with pixel-precise ground truth; private evaluable only through MVTec's server) — preserving the setting where defect types are unknown until deployment.
  3. **Stricter metric:** in addition to AU-PRO(0.30), the paper reports AU-PRO(0.05) — restricting the PRO curve's false-positive-rate range to 0.05<sup>†</sup> to harshly penalize noisy localization.
  4. Published results used as context: EfficientAD ~58.7% AU-PRO(0.30); PatchCore ~53.8%; the seven/eight SOTA methods average **below 60% at AU-PRO(0.30)** and **below ~30% at AU-PRO(0.05)**; PatchCore-class methods fall below 30% on hard categories (e.g. Can, Rice).
- **Why localization metrics matter:** per-pixel AUROC is inflated by extreme class imbalance (anomalies are tiny); the **Per-Region Overlap (PRO)** metric (Bergmann et al.) computes region-scoped recall per connected ground-truth region, averaged across regions, and integrates only over FPR ≤ threshold — so every defect counts equally regardless of size, which a plain pixel AUROC hides.

<sup>†</sup> *AU-PRO(0.05) integrates the PRO curve over the FPR range [0, 0.05] instead of [0, 0.30], i.e. only very precise anomaly maps score well.*

---

## 5. Cross-family summary of objectives

| Family | Objective it optimizes for | The single assumption everything rests on |
|---|---|---|
| Reconstruction (AE/VAE/GAN) | Faithful normal reconstruction | Anomalies are unreconstructable with limited capacity — *known to be false* for small/high-variance defects |
| DRAEM | Discriminative margin via synthetic defects | A boundary learned on pseudo-defects generalizes to real ones |
| Embedding (PaDiM/PatchCore) | Distance to normal-feature distribution | Frozen pretrained features separate normal from defective patches |
| Student–teacher (EfficientAD) | Cheap asymmetry between networks | A student forced to mimic normals only cannot mimic anomalies |
| Density (FastFlow/CFLOW) | Log-density of normal features | Normal features are well-modeled by a flow to a simple base distribution |

Every family solves the *same* cold-start problem; the differences are in *what the notion of "normality" is* and *which failure mode it trades away*.

---

## Key references

- Heckler-Kram, Neudeck, Scheler, König & Steger — *The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection*. IJCV 134(4), 2026. arXiv:2503.21622.
- Bergmann, Fauser, Sattlegger & Steger — *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*. CVPR 2019.
- Bergmann, Batzner, Fauser, Sattlegger & Steger — *Beyond Dents and Scratches: Logical Constraints in Unsupervised Anomaly Detection and Localization* (introduces AU-PRO). IJCV 2022.
- Roth, Pemula, Zepeda, Schölkopf, Brox & Gehler — *Towards Total Recall in Industrial Anomaly Detection* (PatchCore). CVPR 2022. arXiv:2106.08265.
- Defard, Setkov, Loesch & Audigier — *PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization*. ICPR-W 2021. arXiv:2011.08785.
- Batzner, Heckler & König — *EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies*. WACV 2024. arXiv:2303.14535.
- Bergmann et al. — *Uninformed Students: Student–Teacher Anomaly Detection with Discriminative Latent Embeddings*. CVPR 2020.
- Zavrtanik, Kristan & Skočaj — *DRAEM: A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection*. ICCV 2021. arXiv:2108.07610.
- Zhang et al. — *SuperAD: A Training-free Anomaly Classification and Segmentation Method*. CVPR 2025 VAND 3.0 challenge report. arXiv:2505.19750.
- Oquab et al. — *DINOv2: Learning Robust Visual Features without Supervision*. arXiv:2304.07193.