MODEL_PATH = "models/yolo26x.pt"

# Use one of:
# 0                  -> webcam
# "video.mp4"        -> file
# "rtsp://..."       -> RTSP camera
SOURCE = "video.mp4"
SAVE_OUTPUT = True
OUTPUT_PATH = "output.mp4"
DISPLAY = True   # optional

DEVICE = 0

CONF_THRESHOLD = 0.55
IOU_THRESHOLD = 0.45

# Capture queue size; keep small for low latency
QUEUE_SIZE = 2

# Vehicle classes to keep
TARGET_CLASSES = {
    "car",
    "motorcycle",
    "bus",
}

# Tracker settings
MAX_MISSED_FRAMES = 20

# Logging
LOG_JSONL_PATH = "tracks.jsonl"
LOG_CSV_PATH = "tracks.csv"