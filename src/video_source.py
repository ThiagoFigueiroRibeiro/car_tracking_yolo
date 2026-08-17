import cv2
import time
import threading
from queue import Queue, Full
from src.types import FramePacket


class VideoSource:
    def __init__(self, source, frame_queue: Queue, drop_old_frames: bool = True):
        self.source = source
        self.frame_queue = frame_queue
        self.cap = None
        self.running = False
        self.thread = None
        self.frame_id = 0
        self.fps = 30.0
        self.drop_old_frames = drop_old_frames

    def start(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {self.source}")

        cap_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if cap_fps and cap_fps > 1:
            self.fps = cap_fps

        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _push(self, item):
        if not self.drop_old_frames:
            self.frame_queue.put(item)
            return

        try:
            self.frame_queue.put_nowait(item)
        except Full:
            try:
                _ = self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(item)
            except Full:
                pass

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            packet = FramePacket(
                frame_id=self.frame_id,
                timestamp=time.time(),
                frame=frame,
            )
            self.frame_id += 1
            self._push(packet)

        self._push(None)
        self.running = False

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()