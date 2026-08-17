from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from src.types import TrackedObject


@dataclass
class TrackState:
    track_id: int
    class_name: str
    confidence: float

    bbox_xyxy: np.ndarray
    polygon: Optional[np.ndarray]

    first_seen: float
    last_seen: float
    hits: int = 1
    missed: int = 0

    prev_center: Optional[np.ndarray] = None
    last_center: Optional[np.ndarray] = None

    motion_prev_center: Optional[np.ndarray] = None
    motion_last_center: Optional[np.ndarray] = None

    heading_dir: Optional[np.ndarray] = None
    stationary_frames: int = 0
    is_moving: bool = False

    smooth_center: Optional[np.ndarray] = None
    smooth_wh: Optional[np.ndarray] = None

    # stop tracking
    stop_start_time: Optional[float] = None
    total_stopped_time: float = 0.0
    stop_events: int = 0


class TrackManager:
    def __init__(
        self,
        max_missed_frames: int = 20,
        motion_eps: float = 0.1,
        ema_alpha: float = 0.35,
        stop_after_frames: int = 6,
    ):
        self.max_missed_frames = max_missed_frames
        self.motion_eps = motion_eps
        self.ema_alpha = ema_alpha
        self.stop_after_frames = stop_after_frames
        self.active: Dict[int, TrackState] = {}

    @staticmethod
    def _bbox_center_wh(bbox_xyxy: np.ndarray):
        x1, y1, x2, y2 = bbox_xyxy
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        return (
            np.array([cx, cy], dtype=np.float32),
            np.array([w, h], dtype=np.float32),
        )

    @staticmethod
    def _cxcywh_to_xyxy(center: np.ndarray, wh: np.ndarray) -> np.ndarray:
        cx, cy = center
        w, h = wh
        return np.array(
            [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
            dtype=np.float32,
        )

    @staticmethod
    def _normalize(vec: np.ndarray) -> Optional[np.ndarray]:
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            return None
        return (vec / norm).astype(np.float32)

    def _ema(self, prev: np.ndarray, new: np.ndarray) -> np.ndarray:
        return self.ema_alpha * new + (1.0 - self.ema_alpha) * prev

    def _close_stop_interval(self, state: TrackState, timestamp: float):
        if state.stop_start_time is not None:
            state.total_stopped_time += max(0.0, timestamp - state.stop_start_time)
            state.stop_start_time = None

    def update(self, tracked_objects: List[TrackedObject], timestamp: float, compensated_centers=None):
        seen_ids = set()
        new_tracks = []
        removed_tracks = []

        for obj in tracked_objects:
            seen_ids.add(obj.track_id)

            raw_center, raw_wh = self._bbox_center_wh(obj.bbox_xyxy)

            motion_center = raw_center
            if compensated_centers is not None and obj.track_id in compensated_centers:
                motion_center = np.asarray(compensated_centers[obj.track_id], dtype=np.float32)

            if obj.track_id not in self.active:
                smoothed_center = raw_center.copy()
                smoothed_wh = raw_wh.copy()
                smoothed_bbox = self._cxcywh_to_xyxy(smoothed_center, smoothed_wh)

                state = TrackState(
                    track_id=obj.track_id,
                    class_name=obj.class_name,
                    confidence=obj.confidence,
                    bbox_xyxy=smoothed_bbox,
                    polygon=obj.polygon,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    prev_center=None,
                    last_center=smoothed_center.copy(),
                    motion_prev_center=None,
                    motion_last_center=motion_center.copy(),
                    heading_dir=None,
                    stationary_frames=0,
                    is_moving=False,
                    smooth_center=smoothed_center,
                    smooth_wh=smoothed_wh,
                    stop_start_time=timestamp,  # start as stopped until motion is confirmed
                    total_stopped_time=0.0,
                    stop_events=0,
                )
                self.active[obj.track_id] = state
                new_tracks.append(state)
                continue

            state = self.active[obj.track_id]

            # motion logic using compensated coordinates
            was_moving = state.is_moving

            if state.motion_last_center is not None:
                delta = motion_center - state.motion_last_center
                dist = float(np.linalg.norm(delta))

                move_thresh = max(self.motion_eps, 0.01 * max(raw_wh[0], raw_wh[1]))
                stop_thresh = max(self.motion_eps * 0.5, 0.005 * max(raw_wh[0], raw_wh[1]))

                if dist >= move_thresh:
                    new_dir = self._normalize(delta)
                    if new_dir is not None:
                        state.heading_dir = new_dir
                    state.stationary_frames = 0
                    state.is_moving = True
                elif dist < stop_thresh:
                    state.stationary_frames += 1
                    if state.stationary_frames >= self.stop_after_frames:
                        state.is_moving = False
            else:
                state.stationary_frames += 1
                if state.stationary_frames >= self.stop_after_frames:
                    state.is_moving = False

            if (not was_moving) and state.is_moving:
                self._close_stop_interval(state, timestamp)
            elif was_moving and (not state.is_moving):
                state.stop_start_time = timestamp
                state.stop_events += 1

            state.motion_prev_center = state.motion_last_center
            state.motion_last_center = motion_center.copy()

            # EMA smoothing for display bbox
            if state.smooth_center is None:
                state.smooth_center = raw_center.copy()
            else:
                state.smooth_center = self._ema(state.smooth_center, raw_center)

            if state.smooth_wh is None:
                state.smooth_wh = raw_wh.copy()
            else:
                state.smooth_wh = self._ema(state.smooth_wh, raw_wh)

            smoothed_bbox = self._cxcywh_to_xyxy(state.smooth_center, state.smooth_wh)

            state.prev_center = state.last_center
            state.last_center = state.smooth_center.copy()
            state.class_name = obj.class_name
            state.confidence = obj.confidence
            state.bbox_xyxy = smoothed_bbox
            state.polygon = obj.polygon
            state.last_seen = timestamp
            state.hits += 1
            state.missed = 0

        for track_id in list(self.active.keys()):
            if track_id not in seen_ids:
                state = self.active[track_id]
                state.missed += 1

                if state.missed > self.max_missed_frames:
                    # if track is removed while stopped, close interval
                    self._close_stop_interval(state, timestamp)
                    removed_tracks.append(self.active.pop(track_id))

        return new_tracks, removed_tracks

    def snapshot(self):
        return list(self.active.values())

    def get_track(self, track_id: int) -> Optional[TrackState]:
        return self.active.get(track_id)

    def clear(self):
        self.active.clear()

    def average_stopped_time(self) -> float:
        """
        Average total stopped time across tracks that have at least one stop event.
        If a track is currently stopped, include the open interval up to now only if you
        pass a timestamp externally; otherwise this returns only finalized time.
        """
        stopped_tracks = [t for t in self.active.values() if t.stop_events > 0 or t.total_stopped_time > 0.0]
        if not stopped_tracks:
            return 0.0
        return float(np.mean([t.total_stopped_time for t in stopped_tracks]))