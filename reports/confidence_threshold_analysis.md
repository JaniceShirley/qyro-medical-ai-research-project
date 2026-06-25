# Confidence Threshold Analysis

## Objective
Determine the clinically optimal operating point for bounding box confidence. Balancing Precision (avoiding false positives on normal skin) and Recall (detecting as many true acne lesions as possible) is critical for clinical adoption. 

## Sweep Results

| Confidence Threshold | Precision | Recall | mAP50 | F1 Score |
|----------------------|-----------|--------|-------|----------|
| 0.10                 | 0.6875    | 0.6480 | 0.7029| 0.6672   |
| 0.15                 | 0.6875    | 0.6480 | 0.6989| 0.6672   |
| 0.20                 | 0.6875    | 0.6480 | 0.6916| 0.6672   |
| **0.25**             | **0.6857**| **0.6498** | **0.6827** | **0.6672** |
| 0.30                 | 0.7023    | 0.6428 | 0.6820| 0.6712   |
| 0.35                 | 0.7421    | 0.5930 | 0.6715| 0.6592   |
| 0.40                 | 0.7768    | 0.5441 | 0.6606| 0.6400   |

*(Note: Baseline NMS IoU used was 0.60).*

## Observations
1. **Precision-Recall Tradeoff Pivot**: At confidence `0.25`, the model achieves its maximum Recall of roughly 0.65 (0.6498). Pushing confidence higher (e.g., to 0.30) raises Precision over 0.70 but causes Recall to drop to 0.6428. 
2. **Low-Confidence Stagnation**: Dropping confidence below `0.25` (down to `0.10`) does not increase Recall further. In fact, Recall slightly drops to `0.6480` due to NMS box limits and lower-confidence false positives suppressing nearby true positives.
3. **Conclusion**: The clinically optimal confidence threshold is `0.25`. This is the exact pivot point that maximizes Recall without severely degrading Precision or mAP50 below the acceptable Phase 7 target thresholds.
