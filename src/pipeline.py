import os
import cv2
import time
from queue import Queue, Empty

from src.video_source import VideoSource
from src.detector import YOLODetector
from src.filter import VehicleFilter
from src.tracker import ByteTrackTracker
from src.track_manager import TrackManager
from src.annotator import Annotator
from src.logger import TrackLogger
from src.event_engine import EventEngine
from src.camera_motion import CameraMotionEstimator


class VehicleOBBPipeline:
    def __init__(
        self,
        source,
        model_path,
        target_classes,
        conf_threshold=0.35,
        iou_threshold=0.45,
        queue_size=2,
        max_missed_frames=20,
        display=True,
        save_output=False,
        output_path="output.mp4",
        log_jsonl_path=None,
        log_csv_path=None,
    ):
        self.frame_queue = Queue(maxsize=queue_size)

        is_file = isinstance(source, str) and os.path.isfile(source)

        self.source = VideoSource(
            source,
            self.frame_queue,
            drop_old_frames=not is_file,
        )

        self.detector = YOLODetector(
            model_path=model_path,
            conf=conf_threshold,
            iou=iou_threshold,
            device=0,
            imgsz=960,
        )
        self.vehicle_filter = VehicleFilter(target_classes, conf_threshold=conf_threshold)
        self.tracker = ByteTrackTracker()

        self.track_manager = TrackManager(
            max_missed_frames=max_missed_frames,
            motion_eps=0.35,
            ema_alpha=0.35,
            stop_after_frames=20,
        )

        self.camera_motion = CameraMotionEstimator()

        self.annotator = Annotator()
        self.logger = TrackLogger(jsonl_path=log_jsonl_path, csv_path=log_csv_path)
        self.event_engine = EventEngine()

        self.display = display
        self.save_output = save_output
        self.output_path = output_path

        self.running = False
        self.writer = None

    def _init_writer(self, frame):
        if not self.save_output or self.writer is not None:
            return

        h, w = frame.shape[:2]
        fps = self.source.fps if self.source.fps > 1 else 30.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.output_path, fourcc, fps, (w, h))

        if not self.writer.isOpened():
            raise RuntimeError(f"Unable to open video writer: {self.output_path}")

    @staticmethod
    def _bbox_center_xyxy(bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy
        return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]

    def run(self):
        self.running = True
        self.source.start()

        try:
            while self.running:
                try:
                    packet = self.frame_queue.get(timeout=1.0)
                except Empty:
                    if not self.source.running:
                        break
                    continue

                if packet is None:
                    break

                frame = packet.frame
                frame_id = packet.frame_id
                timestamp = packet.timestamp

                self._init_writer(frame)

                # 1) Estimate camera motion on the raw frame
                global_warp, cam_info = self.camera_motion.update(frame)

                t0 = time.time()

                # 2) Detect / filter / track
                detections = self.detector.detect(frame, frame_id)
                detections = self.vehicle_filter.apply(detections)
                tracked_objects = self.tracker.update(detections, frame)

                # 3) Build compensated centers in stable reference coordinates
                compensated_centers = {}
                for obj in tracked_objects:
                    raw_center = self._bbox_center_xyxy(obj.bbox_xyxy)
                    compensated_centers[obj.track_id] = self.camera_motion.transform_point(
                        raw_center,
                        global_warp,
                    )

                # 4) Update TrackManager with compensated centers
                new_tracks, removed_tracks = self.track_manager.update(
                    tracked_objects,
                    timestamp,
                    compensated_centers=compensated_centers,
                )

                events = self.event_engine.process(new_tracks, removed_tracks)
                for e in events:
                    print(e)

                self.logger.log_tracks(timestamp, frame_id, tracked_objects)

                fps = 1.0 / max(time.time() - t0, 1e-6)

                avg_stop = self.track_manager.average_stopped_time()

                # 5) Draw smoothed bboxes + triangle/colors using TrackManager state
                annotated = self.annotator.draw(
                    frame,
                    self.track_manager.snapshot(),
                    fps=fps,
                )

                # Optional debug overlay for camera motion
                cv2.putText(
                    annotated,
                    f"cam_motion: {cam_info['method']} | matches={cam_info['matches']} | inliers={cam_info['inliers']}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    annotated,
                    f"Avg stopped time: {avg_stop:.1f}s",
                    (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                

                if self.writer is not None:
                    self.writer.write(annotated)

                if self.display:
                    cv2.imshow("Vehicle OBB Tracking", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        finally:
            self.stop()

    def stop(self):
        self.running = False
        self.source.stop()

        if self.writer is not None:
            self.writer.release()
            self.writer = None

        self.logger.close()
        cv2.destroyAllWindows()