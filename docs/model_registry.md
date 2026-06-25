# QYRO Model Registry

This registry catalogs all official checkpoint models, their corresponding datasets, evaluated metrics, and current production status.

| Model Checkpoint Alias | Dataset Version | mAP50 | Precision | Recall | Status |
|------------------------|-----------------|-------|-----------|--------|--------|
| YOLOv8n Baseline | QYRO Dataset v1 | 0.6354 | 0.6388 | 0.6225 | Archived |
| YOLOv8s Baseline | QYRO Dataset v1 | 0.4309 | 0.4620 | 0.4508 | Failed (Undertrained) |
| YOLOv8s Convergence | QYRO Dataset v1 | 0.6940 | 0.6860 | 0.6400 | Archived Base |
| **YOLOv8s Production** | **QYRO Dataset v1** | **0.6827** | **0.6857** | **0.6498** | **Active Production** |

*(Note: YOLOv8s Production metrics reflect inference at optimal confidence threshold of `0.25` and IoU `0.60`).*
