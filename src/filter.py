class VehicleFilter:
    def __init__(self, target_classes, conf_threshold: float = 0.35):
        self.target_classes = set(target_classes)
        self.conf_threshold = conf_threshold

    def apply(self, detections):
        filtered = []
        for det in detections:
            if det.confidence < self.conf_threshold:
                continue
            if det.class_name not in self.target_classes:
                continue
            filtered.append(det)
        return filtered