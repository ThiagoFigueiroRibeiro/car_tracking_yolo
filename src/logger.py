import csv
import json
from pathlib import Path


class TrackLogger:
    def __init__(self, jsonl_path: str | None = None, csv_path: str | None = None):
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        self.csv_path = Path(csv_path) if csv_path else None

        self._csv_file = None
        self._csv_writer = None

        if self.csv_path:
            self._csv_file = open(self.csv_path, "a", newline="", buffering=1)
            self._csv_writer = csv.writer(self._csv_file)
            if self.csv_path.stat().st_size == 0:
                self._csv_writer.writerow([
                    "timestamp",
                    "frame_id",
                    "track_id",
                    "class_name",
                    "confidence",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                ])

    def log_tracks(self, timestamp: float, frame_id: int, tracks):
        record = {
            "timestamp": timestamp,
            "frame_id": frame_id,
            "tracks": [],
        }

        for t in tracks:
            x1, y1, x2, y2 = map(float, t.bbox_xyxy)
            item = {
                "track_id": int(t.track_id),
                "class_name": t.class_name,
                "confidence": float(t.confidence),
                "bbox_xyxy": [x1, y1, x2, y2],
            }
            record["tracks"].append(item)

            if self._csv_writer:
                self._csv_writer.writerow([
                    timestamp,
                    frame_id,
                    t.track_id,
                    t.class_name,
                    t.confidence,
                    x1,
                    y1,
                    x2,
                    y2,
                ])

        if self.jsonl_path:
            with open(self.jsonl_path, "a") as f:
                f.write(json.dumps(record) + "\n")

    def close(self):
        if self._csv_file:
            self._csv_file.close()