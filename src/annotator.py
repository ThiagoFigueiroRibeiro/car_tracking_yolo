import cv2
import numpy as np

GREEN = (0, 255, 0)
RED = (0, 0, 255)


def color_for_state(track):
    return GREEN if getattr(track, "is_moving", False) else RED


class Annotator:
    def draw(self, frame, tracks, fps: float | None = None, avg_stop: float | None = None):
        out = frame.copy()

        for t in tracks:
            color = color_for_state(t)
            x1, y1, x2, y2 = map(int, t.bbox_xyxy)

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            label = f"ID {t.track_id} | {t.class_name}"
            cv2.putText(
                out,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

            # triangle logic can remain if you still want it
            # (draw from heading_dir / motion state)

        if fps is not None:
            cv2.putText(
                out,
                f"FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        if avg_stop is not None:
            cv2.putText(
                out,
                f"Avg stopped time: {avg_stop:.1f}s",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return out