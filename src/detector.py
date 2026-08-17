from ultralytics import YOLO
from src.types import Detection


class YOLODetector:
    def __init__(self, model_path: str, conf: float = 0.35, iou: float = 0.45, device=0, imgsz=960):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.device = device
        self.imgsz = imgsz

    def detect(self, frame, frame_id: int):
        result = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
            device=self.device,
        )[0]

        detections = []
        names = result.names

        if result.boxes is None:
            return detections

        boxes = result.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            class_name = names[cls_id]
            confidence = float(boxes.conf[i].item())
            xyxy = boxes.xyxy[i].cpu().numpy().astype("float32")

            detections.append(
                Detection(
                    frame_id=frame_id,
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=confidence,
                    xyxy=xyxy,
                    polygon=None,
                )
            )

        return detections