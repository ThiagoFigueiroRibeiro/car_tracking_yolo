from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class FramePacket:
    frame_id: int
    timestamp: float
    frame: np.ndarray


@dataclass
class Detection:
    frame_id: int
    class_id: int
    class_name: str
    confidence: float
    xyxy: np.ndarray              # [x1, y1, x2, y2]
    polygon: Optional[np.ndarray]  # 4x2 points for OBB


@dataclass
class TrackedObject:
    track_id: int
    class_name: str
    confidence: float
    bbox_xyxy: np.ndarray         # [x1, y1, x2, y2]
    polygon: Optional[np.ndarray]  # OBB polygon if available