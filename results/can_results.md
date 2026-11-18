# ScanGuard results — can

Mode: **quick slice**, input 256x256, device `cpu`.
Train / val / test: 64 / 16 / 48 images (24 anomalous).

| Metric | Autoencoder | PaDiM |
|---|---:|---:|
| Image AUROC | 0.4271 | 0.4635 |
| Image AUPR | 0.5028 | 0.4537 |
| Image precision | 0.2941 | 0.3750 |
| Image recall | 0.2083 | 0.2500 |
| Image F1 | 0.2439 | 0.3000 |
| Pixel AUROC | 0.6170 | 0.6789 |
| AU-PRO (0.30) | 0.1207 | 0.2070 |
| AU-PRO (0.05) | 0.0170 | 0.0767 |

Thresholds: autoencoder 10.9186, PaDiM 77.3383 (99th percentile of defect-free validation scores).
