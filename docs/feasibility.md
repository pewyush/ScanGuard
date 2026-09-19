# Feasibility Assessment

*Delivery artifact for the "Review of Existing Systems & Project Feasibility" check.*

Honest assessment of whether the project is doable in the available time/resources: what data we will use, whether it is accessible, what compute it needs, and what could realistically go wrong.

## Data

- **Dataset:** MVTec AD 2 (MVTec Software GmbH) — the dataset already chosen in the README.
- **Contents:** ~8,000+ high-resolution images across 8 industrial inspection scenarios, each with a defect-free train/val split and two test splits (one with public pixel-precise ground truth, one private).
- **Why it fits:** the anomaly-detection framing matches the dataset's intended unsupervised setup exactly; the hard scenarios give room to compare methods rather than reproduce saturated scores.

### Accessibility

| Requirement | Status |
|---|---|
| Free to download | Yes — short form on mvtec.com, no payment |
| License permits coursework use | Yes — CC BY-NC-SA 4.0 (non-commercial / academic) |
| Ground truth locally available | Yes — public split ships pixel-precise GT |
| Code utilities | Yes — official PyTorch data-loading + submission utils published by MVTec |
| External dependencies | Benchmark server (mvtec.com/benchmark) for the private split only — optional |

**Residual risk (low–medium):** the download is gated by an approval form, so there is some lead time; the private-split evaluation depends on an external server, which is why it is treated as optional.

## Compute

All three method families run on a single consumer GPU:

| Approach | Training | GPU footprint |
|---|---|---|
| Convolutional autoencoder (baseline) | Full training, small model | Low |
| PatchCore-style embedding | No training (feature extraction + memory bank) | Low–moderate; inference cost scales with bank size (kNN) |
| EfficientAD-style student–teacher | Modest training | Moderate |

**Constraint:** running all 8 categories is a time/compute risk. The plan therefore commits to **1–3 scenario categories**, balancing defect-type diversity against cost — consistent with README plan step 1.

## Timeline of the main risks (what could realistically go wrong)

1. **Reconstruction baseline performs poorly** — the strongest risk. AEs often *reconstruct defects too well*, and on MVTec AD 2's small/transparent/high-variance defects the baseline numbers may be near-unusable. This is expected and well documented, but the write-up must frame it as a methodology demonstration, not a failure, and the baseline must not be over-interpreted.
2. **Categories chosen may yield little signal** — the hardest scenarios (transparent surfaces, small defects) push even PatchCore below ~30% AU-PRO. Choosing too many hard categories could leave a write-up with nothing to discuss; preference is for a mix of one "reasonable" and one "hard" category.
3. **ImageNet-feature domain bias** — pretrained backbones are misaligned with industrial imagery (a known false-detection failure mode); relevant when comparing against the reconstruction baseline and listed as a limitation/future work.
4. **External dependencies** — download-approval lead time; benchmark-server availability if the private split is attempted; no access to a GPU at crunch time would force downsizing the experiment matrix (a realistic fallback is a smaller autoencoder + one embedding method on fewer categories).
5. **Scope creep** — deployment/serving is already explicitly out of scope in the README; any temptation to add MLOps tooling is a time sink and is excluded.

## Bottom line

The project is **feasible within scope** provided (a) geometry is limited to 1–3 categories, (b) the autoencoder is positioned as a deliberate weak baseline rather than a target, and (c) external dependencies (data form, benchmark server) are kicked off early. The honest deliverable is a methodology comparison on an unsaturated benchmark — not a SOTA claim.

## Fallback plans

- **If baseline collapses to ~random on all chosen categories:** switch the baseline to a VAE or a small memory-based method (PaDiM) instead of a raw AE; keep the discussion of why image-level AE fails.
- **If GPU time runs out:** pre-extract features for the embedding method (cheap), and compare only image-level AUROC instead of pixel-level localization.
- **If localization metrics (AU-PRO) prove unwieldy:** fall back to pixel-AUROC, which is simpler to implement, and report AU-PRO only if the official utils are used as planned.