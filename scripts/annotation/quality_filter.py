import numpy as np

def filter_bounding_boxes(boxes_data, min_size=0.005, max_area=0.80, min_conf=0.25):
    """
    boxes_data: list of dicts: {'cls': 0, 'cx': float, 'cy': float, 'w': float, 'h': float, 'conf': float}
    returns: (accepted_boxes, rejected_boxes_count)
    """
    accepted = []
    rejected_count = 0

    for box in boxes_data:
        cx, cy, w, h, conf = box['cx'], box['cy'], box['w'], box['h'], box['conf']

        # 1. Bounds check
        if cx < 0.0 or cx > 1.0 or cy < 0.0 or cy > 1.0:
            rejected_count += 1
            continue

        # 2. Size check
        if w < min_size or h < min_size:
            rejected_count += 1
            continue

        # 3. Max area check
        if (w * h) > max_area:
            rejected_count += 1
            continue

        # 4. Confidence check
        if conf < min_conf:
            rejected_count += 1
            continue

        accepted.append(box)

    return accepted, rejected_count
