import numpy as np
import supervision as sv

from src.types import Detection, TrackedObject


class ByteTrackTracker:
    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
    ):
        """
        ByteTrack wrapper using supervision.

        Args:
            track_activation_threshold: minimum confidence to activate a track
            lost_track_buffer: how many frames a track can remain lost
            minimum_matching_threshold: IoU matching threshold
            frame_rate: expected input frame rate
        """
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )

    def update(self, detections, frame=None):
        """
        Update tracker with detection list.

        Args:
            detections: list[Detection]
            frame: unused, kept for pipeline compatibility

        Returns:
            list[TrackedObject]
        """
        if not detections:
            return []

        xyxy = []
        confidence = []
        class_id = []

        for det in detections:
            xyxy.append(det.xyxy)
            confidence.append(det.confidence)
            class_id.append(det.class_id)

        xyxy = np.asarray(xyxy, dtype=np.float32)
        confidence = np.asarray(confidence, dtype=np.float32)
        class_id = np.asarray(class_id, dtype=np.int32)

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )

        tracked = self.tracker.update_with_detections(sv_detections)

        tracked_objects = []
        tracker_ids = tracked.tracker_id

        if tracker_ids is None:
            return tracked_objects

        for i in range(len(tracked)):
            track_id = tracker_ids[i]
            if track_id is None:
                continue

            cls_id = int(tracked.class_id[i])
            conf = float(tracked.confidence[i])
            box = tracked.xyxy[i].astype(np.float32)

            # Best-effort matching back to original detection so we keep polygon
            matched_det = self._match_detection(box, detections)

            tracked_objects.append(
                TrackedObject(
                    track_id=int(track_id),
                    class_name=matched_det.class_name if matched_det else str(cls_id),
                    confidence=conf,
                    bbox_xyxy=box,
                    polygon=matched_det.polygon if matched_det else None,
                )
            )

        return tracked_objects

    def _match_detection(self, track_box, detections, iou_thresh: float = 0.2):
        best_det = None
        best_iou = 0.0

        for det in detections:
            score = self._iou_xyxy(track_box, det.xyxy)
            if score > best_iou:
                best_iou = score
                best_det = det

        return best_det if best_iou >= iou_thresh else None

    @staticmethod
    def _iou_xyxy(a, b):
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