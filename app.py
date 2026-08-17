from config import (
    MODEL_PATH,
    SOURCE,
    TARGET_CLASSES,
    CONF_THRESHOLD,
    IOU_THRESHOLD,
    QUEUE_SIZE,
    MAX_MISSED_FRAMES,
    DISPLAY,
    SAVE_OUTPUT,
    OUTPUT_PATH,
    LOG_JSONL_PATH,
    LOG_CSV_PATH,
)
from src.pipeline import VehicleOBBPipeline


def main():
    pipeline = VehicleOBBPipeline(
        source=SOURCE,
        model_path=MODEL_PATH,
        target_classes=TARGET_CLASSES,
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        queue_size=QUEUE_SIZE,
        max_missed_frames=MAX_MISSED_FRAMES,
        display=DISPLAY,
        save_output=SAVE_OUTPUT,
        output_path=OUTPUT_PATH,
        log_jsonl_path=LOG_JSONL_PATH,
        log_csv_path=LOG_CSV_PATH,
    )
    pipeline.run()


if __name__ == "__main__":
    main()