"""
Vehicle Detection Server
- RTSP → HTTP stream conversion via FFmpeg subprocess
- YOLOv8 + SAHI for ambulance/car detection
- Flask API with MJPEG streaming
"""

import cv2
import numpy as np
import subprocess
import threading
import time
import logging
import json
import os
import queue
from datetime import datetime
from flask import Flask, Response, jsonify, send_from_directory
from flask_cors import CORS

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')
CORS(app)

# ─── Configuration ────────────────────────────────────────────────────────────
RTSP_URL = "rtsp://admin:Syscom2026@169.254.18.91:554/ISAPI/Streaming/channels/1"
RTSP_TIMEOUT = 10          # seconds before declaring stream dead
RECONNECT_DELAY = 5        # seconds between reconnect attempts
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 25            # Increased from 15 for smoother video
DETECTION_INTERVAL = 6     # run detection every 6 frames (was 3) for better performance
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45

# Custom class IDs from fine-tuned dataset.yaml
VEHICLE_CLASSES = {0: "ambulance", 1: "bus", 2: "car", 3: "motorcycle", 4: "truck"}

# ─── Global state ─────────────────────────────────────────────────────────────
state = {
    "frame_count": 0,
    "detection_count": {"car": 0, "ambulance": 0, "bus": 0, "truck": 0, "motorcycle": 0},
    "total_detections": 0,
    "stream_status": "connecting",
    "fps": 0,
    "last_detections": [],
    "start_time": time.time(),
}

latest_frame_lock = threading.Lock()
latest_frame = None          # raw JPEG bytes of the annotated frame
frame_queue = queue.Queue(maxsize=2)

# ─── Vehicle tracking (to avoid duplicate detections) ────────────────────────
tracked_vehicles = {}  # Format: {(cx, cy): {"label": str, "timestamp": float, "bbox": [x1,y1,x2,y2]}}
tracking_lock = threading.Lock()
TRACKING_DISTANCE_THRESHOLD = 80  # pixels
TRACKING_TIMEOUT = 5  # seconds

def get_bbox_center(bbox):
    """Calculate center point of bounding box."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def euclidean_distance(p1, p2):
    """Calculate distance between two points."""
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def is_new_vehicle(bbox, label):
    """Check if detection is a new vehicle or a repeat of tracked one."""
    global tracked_vehicles
    current_time = time.time()
    center = get_bbox_center(bbox)
    
    # Clean up old tracks
    expired_keys = [k for k, v in tracked_vehicles.items() 
                    if current_time - v["timestamp"] > TRACKING_TIMEOUT]
    for k in expired_keys:
        del tracked_vehicles[k]
    
    # Check if close to existing vehicle
    for tracked_center, tracked_info in tracked_vehicles.items():
        dist = euclidean_distance(center, tracked_center)
        if dist < TRACKING_DISTANCE_THRESHOLD and tracked_info["label"] == label:
            # Update timestamp but don't report as new
            tracked_info["timestamp"] = current_time
            tracked_info["bbox"] = bbox
            return False
    
    # New vehicle - track it
    tracked_vehicles[center] = {
        "label": label,
        "timestamp": current_time,
        "bbox": bbox
    }
    return True

# ─── YOLO + SAHI Setup ────────────────────────────────────────────────────────
MODEL = None
USE_SAHI = False

def load_model():
    global MODEL, USE_SAHI
    try:
        from ultralytics import YOLO
        import os
        model_path = "runs/ambulance_detection/weights/best.pt"
        if not os.path.exists(model_path):
            logger.warning(f"Custom model '{model_path}' not found, falling back to 'yolov8n.pt'. Make sure to run the training script!")
            model_path = "yolov8n.pt"
            
        logger.info(f"Loading YOLO model '{model_path}'...")
        MODEL = YOLO(model_path)
        logger.info("YOLO model loaded successfully")

        # SAHI disabled by default (too slow for real-time streaming)
        # To enable SAHI: set USE_SAHI = True below
        USE_SAHI = False
        if USE_SAHI:
            try:
                from sahi import AutoDetectionModel
                from sahi.predict import get_sliced_prediction
                logger.info("SAHI available – sliced inference enabled")
            except ImportError:
                USE_SAHI = False
                logger.warning("SAHI not installed – running standard inference")
        else:
            logger.info("SAHI disabled for better performance (standard YOLOv8 only)")

    except ImportError:
        logger.warning("ultralytics not installed – running in DEMO mode (random boxes)")
        MODEL = None

load_model()

# ─── Detection helpers ────────────────────────────────────────────────────────

def run_detection_yolo(frame):
    """Standard YOLO inference - optimized for real-time performance."""
    # Use half precision if available for faster inference
    try:
        results = MODEL(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False, half=True)[0]
    except:
        # Fallback to full precision if half precision fails
        results = MODEL(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)[0]
    
    detections = []
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls not in VEHICLE_CLASSES:
            continue
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = VEHICLE_CLASSES[cls]
        detections.append({"bbox": [x1, y1, x2, y2], "label": label, "conf": conf})
    return detections


def run_detection_sahi(frame):
    """SAHI sliced inference for small objects."""
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    import tempfile, cv2

    # SAHI needs a file or PIL image; write temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp.name, frame)
    tmp.close()

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path="yolov8n.pt",
        confidence_threshold=CONF_THRESHOLD,
        device="cpu",
    )
    result = get_sliced_prediction(
        tmp.name,
        detection_model,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )
    os.unlink(tmp.name)

    detections = []
    for obj in result.object_prediction_list:
        cls_name = obj.category.name.lower()
        # Map class names to our labels
        label_map = {"ambulance": "ambulance", "car": "car", "truck": "truck", "bus": "bus", "motorcycle": "motorcycle"}
        if cls_name not in label_map:
            continue
        label = label_map[cls_name]
        bbox = obj.bbox
        x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
        conf = obj.score.value
        detections.append({"bbox": [x1, y1, x2, y2], "label": label, "conf": conf})
    return detections


def detect(frame):
    if MODEL is None:
        return demo_detections(frame)
    try:
        if USE_SAHI:
            return run_detection_sahi(frame)
        return run_detection_yolo(frame)
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return []


def demo_detections(frame):
    """Fake detections when no model is loaded (development/demo mode)."""
    h, w = frame.shape[:2]
    rng = np.random.default_rng(int(time.time()) // 3)
    n = rng.integers(1, 4)
    labels = ["car", "car", "ambulance", "truck", "bus"]
    dets = []
    for _ in range(n):
        lbl = labels[rng.integers(0, len(labels))]
        x1 = int(rng.uniform(0.05, 0.6) * w)
        y1 = int(rng.uniform(0.1, 0.6) * h)
        x2 = x1 + int(rng.uniform(0.1, 0.3) * w)
        y2 = y1 + int(rng.uniform(0.1, 0.3) * h)
        dets.append({"bbox": [x1, y1, min(x2,w-1), min(y2,h-1)], "label": lbl, "conf": float(rng.uniform(0.6, 0.99))})
    return dets

# ─── Annotation ───────────────────────────────────────────────────────────────

LABEL_COLORS = {
    "car":        (0, 200, 80),
    "ambulance":  (0, 60, 255),
    "truck":      (255, 130, 0),
    "bus":        (0, 180, 255),
    "motorcycle": (200, 0, 200),
}

def draw_detections(frame, detections):
    overlay = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        conf  = det["conf"]
        color = LABEL_COLORS.get(label, (200, 200, 200))

        # Semi-transparent fill
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
        overlay = frame.copy()

        # Solid border
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Corner accent lines
        ll = min(20, (x2-x1)//3, (y2-y1)//3)
        for (sx, sy, dx, dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(frame, (sx,sy), (sx+dx*ll,sy), color, 3)
            cv2.line(frame, (sx,sy), (sx,sy+dy*ll), color, 3)

        # Label pill
        tag = f"{label.upper()}  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        pad = 5
        tx, ty = x1, y1 - th - pad * 2
        if ty < 0:
            ty = y1 + 2
        cv2.rectangle(frame, (tx, ty), (tx+tw+pad*2, ty+th+pad*2), color, -1)
        cv2.putText(frame, tag, (tx+pad, ty+th+pad), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

    # HUD overlay
    ts = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, ts, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1, cv2.LINE_AA)
    mode_tag = "SAHI+YOLOv8" if USE_SAHI else ("YOLOv8" if MODEL else "DEMO")
    cv2.putText(frame, mode_tag, (frame.shape[1]-150, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,200,255), 1, cv2.LINE_AA)
    return frame

# ─── RTSP → frame reader ──────────────────────────────────────────────────────

class RTSPReader(threading.Thread):
    """
    Reads RTSP via FFmpeg subprocess → raw BGR frames.
    This avoids OpenCV's RTSP quirks and works without RTSP libraries.
    """
    def __init__(self):
        super().__init__(daemon=True, name="RTSPReader")
        self._stop_event = threading.Event()
        self.current_frame = None
        self.lock = threading.Lock()

    def stop(self):
        self._stop_event.set()

    def run(self):
        global latest_frame
        while not self._stop_event.is_set():
            state["stream_status"] = "connecting"
            logger.info(f"Connecting to RTSP: {RTSP_URL}")
            try:
                self._read_stream()
            except Exception as e:
                logger.error(f"Stream error: {e}")
                state["stream_status"] = "error"
            if not self._stop_event.is_set():
                logger.info(f"Reconnecting in {RECONNECT_DELAY}s...")
                time.sleep(RECONNECT_DELAY)

    def _read_stream(self):
        global latest_frame
        cmd = [
            "ffmpeg", "-loglevel", "error",
            "-rtsp_transport", "tcp",
            "-fflags", "nobuffer",          # Discard old frames if buffer builds up
            "-flags", "low_delay",          # Low latency mode
            "-i", RTSP_URL,
            "-vf", f"scale={FRAME_WIDTH}:{FRAME_HEIGHT}",
            "-r", str(TARGET_FPS),
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "pipe:1"
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        frame_size = FRAME_WIDTH * FRAME_HEIGHT * 3
        frame_idx = 0
        state["stream_status"] = "live"
        fps_t0 = time.time()
        fps_count = 0

        try:
            while not self._stop_event.is_set():
                raw = proc.stdout.read(frame_size)
                if len(raw) < frame_size:
                    logger.warning("FFmpeg stream ended or incomplete frame")
                    break

                frame = np.frombuffer(raw, dtype=np.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH, 3))
                frame_idx += 1
                fps_count += 1
                state["frame_count"] += 1

                # FPS calculation
                elapsed = time.time() - fps_t0
                if elapsed >= 2.0:
                    state["fps"] = round(fps_count / elapsed, 1)
                    fps_count = 0
                    fps_t0 = time.time()

                # Run detection every N frames
                detections = []
                if frame_idx % DETECTION_INTERVAL == 0:
                    detections = detect(frame)
                    # Filter: only keep new vehicles
                    new_detections = []
                    with tracking_lock:
                        for d in detections:
                            if is_new_vehicle(d["bbox"], d["label"]):
                                new_detections.append(d)
                                state["total_detections"] += 1
                                lbl = d["label"]
                                state["detection_count"][lbl] = state["detection_count"].get(lbl, 0) + 1
                    state["last_detections"] = new_detections
                else:
                    detections = state.get("last_detections", [])

                annotated = draw_detections(frame.copy(), detections)
                _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 65])
                with latest_frame_lock:
                    latest_frame = jpeg.tobytes()

        finally:
            proc.kill()
            proc.wait()


# ─── Fallback: OpenCV RTSP if FFmpeg unavailable ─────────────────────────────

class OpenCVReader(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True, name="OpenCVReader")
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        global latest_frame
        while not self._stop.is_set():
            state["stream_status"] = "connecting"
            logger.info("OpenCV fallback: connecting to RTSP...")
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)

            if not cap.isOpened():
                logger.error("OpenCV: cannot open stream")
                state["stream_status"] = "error"
                time.sleep(RECONNECT_DELAY)
                continue

            state["stream_status"] = "live"
            frame_idx = 0
            fps_t0 = time.time()
            fps_count = 0
            fail_count = 0

            while not self._stop.is_set():
                ret, frame = cap.read()
                if not ret:
                    fail_count += 1
                    if fail_count > 20:
                        logger.warning("Too many read failures")
                        break
                    time.sleep(0.05)
                    continue
                fail_count = 0

                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                frame_idx += 1
                fps_count += 1
                state["frame_count"] += 1

                elapsed = time.time() - fps_t0
                if elapsed >= 2.0:
                    state["fps"] = round(fps_count / elapsed, 1)
                    fps_count = 0
                    fps_t0 = time.time()

                detections = []
                if frame_idx % DETECTION_INTERVAL == 0:
                    detections = detect(frame)
                    # Filter: only keep new vehicles
                    new_detections = []
                    with tracking_lock:
                        for d in detections:
                            if is_new_vehicle(d["bbox"], d["label"]):
                                new_detections.append(d)
                                state["total_detections"] += 1
                                lbl = d["label"]
                                state["detection_count"][lbl] = state["detection_count"].get(lbl, 0) + 1
                    state["last_detections"] = new_detections
                else:
                    detections = state.get("last_detections", [])

                annotated = draw_detections(frame.copy(), detections)
                _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 65])
                with latest_frame_lock:
                    latest_frame = jpeg.tobytes()

            cap.release()
            time.sleep(RECONNECT_DELAY)


# ─── MJPEG stream endpoint ────────────────────────────────────────────────────

def generate_mjpeg():
    BLANK = _make_blank_frame()
    last_frame = None
    skip_count = 0
    
    while True:
        with latest_frame_lock:
            frame = latest_frame

        if frame is None:
            frame = BLANK
        elif frame == last_frame:
            # Skip duplicate frames to avoid blocking
            skip_count += 1
            if skip_count < 2:
                time.sleep(0.01)
                continue
            skip_count = 0
        else:
            last_frame = frame
            skip_count = 0

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


def _make_blank_frame():
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(img, "Connecting to camera...", (400, 360),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (80, 80, 80), 2, cv2.LINE_AA)
    _, jpeg = cv2.imencode(".jpg", img)
    return jpeg.tobytes()


@app.route("/stream")
def stream():
    return Response(generate_mjpeg(),
                    mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-cache", "Access-Control-Allow-Origin": "*"})


@app.route("/status")
def status():
    uptime = int(time.time() - state["start_time"])
    return jsonify({
        **state,
        "uptime": uptime,
        "model": "SAHI+YOLOv8n" if USE_SAHI else ("YOLOv8n" if MODEL else "DEMO"),
    })


@app.route("/reset", methods=["POST"])
def reset():
    """Reset all detection counters to start fresh."""
    global state
    state["detection_count"] = {"car": 0, "ambulance": 0, "bus": 0, "truck": 0, "motorcycle": 0}
    state["total_detections"] = 0
    state["last_detections"] = []
    state["start_time"] = time.time()
    tracked_vehicles.clear()
    return jsonify({"status": "reset", "message": "Counters reset successfully"})


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# ─── Main ──────────────────────────────────────────────────────────────────────

def start_reader():
    """Try FFmpeg-based reader first, fall back to OpenCV."""
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=3)
        if result.returncode == 0:
            logger.info("FFmpeg found – using RTSPReader")
            reader = RTSPReader()
            reader.start()
            return reader
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    logger.info("FFmpeg not found – using OpenCV fallback reader")
    reader = OpenCVReader()
    reader.start()
    return reader


if __name__ == "__main__":
    reader = start_reader()
    logger.info("Starting Flask server on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
