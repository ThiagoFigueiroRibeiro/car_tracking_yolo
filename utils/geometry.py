import cv2
import numpy as np


def polygon_to_xyxy(polygon: np.ndarray) -> np.ndarray:
    """
    polygon: shape (4, 2)
    returns [x1, y1, x2, y2]
    """
    x1 = float(np.min(polygon[:, 0]))
    y1 = float(np.min(polygon[:, 1]))
    x2 = float(np.max(polygon[:, 0]))
    y2 = float(np.max(polygon[:, 1]))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def xywhr_to_polygon(xywhr: np.ndarray) -> np.ndarray:
    """
    Convert Ultralytics xywhr -> 4 point polygon.

    xywhr = [x_center, y_center, width, height, rotation]
    Rotation may be in radians or degrees depending on version.
    """
    x, y, w, h, r = xywhr.tolist()

    # Heuristic: if angle is small, treat as radians and convert to degrees
    angle_deg = np.degrees(r) if abs(r) <= 2 * np.pi else r

    rect = ((float(x), float(y)), (float(w), float(h)), float(angle_deg))
    pts = cv2.boxPoints(rect)  # shape (4, 2)
    return pts.astype(np.float32)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """
    Intersection over Union for axis-aligned boxes.
    a, b: [x1, y1, x2, y2]
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area
    return float(inter_area / union) if union > 0 else 0.0