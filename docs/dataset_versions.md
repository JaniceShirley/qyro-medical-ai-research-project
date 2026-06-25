# QYRO Dataset Versions

This document tracks the frozen, read-only baseline versions of the QYRO datasets.

## QYRO Dataset v1 (`acne_v1_original`)

**Description**: The original aggregated and cleaned dataset used to train the official QYRO Acne v1 production detector.
**State**: Frozen (Read-Only)

### Statistics
- **Total Images**: 834
- **Splits**: 
  - Train: 681 images
  - Validation: 78 images
  - Test: 75 images
- **Classes**: 1 (`acne`)

### Production Details
- **Production Detector Model**: YOLOv8s Convergence Baseline (`qyro_acne_v1_best.pt`)
- **Inference Thresholds**: 
  - Confidence (`conf`): 0.25
  - NMS IoU (`iou`): 0.60
- **Image Size (`imgsz`)**: 640

*Note: All future modifications, dataset additions, and annotation refinements occur in v2 onward.*
