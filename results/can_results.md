# ScanGuard results — can

Mode: **full data**, input 256x256, device `cpu`.
Train / val / test: 412 / 46 / 162 images (90 anomalous).

| Metric | Autoencoder | PaDiM |
|---|---:|---:|
| Image AUROC | 0.5252 | 0.4938 |
| Image AUPR | 0.5548 | 0.5513 |
| Image precision | 0.5625 | 0.5517 |
| Image recall | 0.4000 | 0.5333 |
| Image F1 | 0.4675 | 0.5424 |
| Pixel AUROC | 0.6926 | 0.7081 |
| AU-PRO (0.30) | 0.2793 | 0.2272 |
| AU-PRO (0.05) | 0.0654 | 0.0288 |

Thresholds: autoencoder 8.4895, PaDiM 48.2573 (99th percentile of defect-free validation scores).
