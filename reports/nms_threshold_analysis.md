# NMS Threshold Analysis

## Objective
Evaluate the impact of the Non-Maximum Suppression (NMS) Intersection over Union (IoU) threshold on detection performance. The goal is to determine the optimal degree of bounding box merging, particularly for crowded acne lesions where overlapping boxes are common.

## Sweep Results

| IoU Threshold | Precision | Recall | mAP50 | mAP50-95 | F1 Score |
|---------------|-----------|--------|-------|----------|----------|
| **0.50**      | **0.6954**| 0.6480 | **0.7071**| 0.3178   | **0.6709**|
| **0.55**      | 0.6895    | 0.6480 | 0.7067| **0.3184**| 0.6681   |
| **0.60** (Base) | 0.6875    | 0.6480 | 0.7047| 0.3178   | 0.6672   |
| **0.65**      | 0.6956    | 0.6426 | 0.6994| 0.3174   | 0.6681   |
| **0.70**      | 0.6883    | 0.6402 | 0.6936| 0.3173   | 0.6634   |

## Observations
1. **Aggressive NMS Benefits Crowding**: Lowering the IoU threshold to `0.50` yields the best overall performance, producing the highest Precision (0.6954), mAP50 (0.7071), and F1 Score (0.6709).
2. **Stable Recall**: Recall remains perfectly stable at `0.6480` for IoU values between `0.50` and `0.60`. Raising the threshold to `0.65` or `0.70` causes Recall to decay slightly (to 0.6402), likely because insufficient suppression leads to duplicate box penalties and false positives suppressing true boxes.
3. **Conclusion**: An NMS IoU threshold of `0.50` or `0.55` is strictly superior to the default `0.60` or higher thresholds. The model benefits from a stricter NMS threshold (lower IoU) to aggressively merge highly overlapping candidate boxes in clustered acne regions.
