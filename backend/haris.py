import cv2
import mediapipe as mp
import os
import urllib.request
import math
import time
import threading

from datetime import datetime
from ultralytics import YOLO
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from app import db, Photo, app

with app.app_context():
    db.create_all()

# --- 1. CONFIGURATION & MODEL PATHS ---
YOLO_MODEL_NAME = "yolov8n.pt"
POSE_MODEL_PATH = "pose_landmarker_full.task"
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"

PROXIMITY_RATIO = 0.15
HOLD_LABEL = "Object Detected Near Person"
LAST_CAPTURE_TIME = 0
COOLDOWN_SECONDS = 3.0
POCKET_THRESHOLD = 0.12

# FIX: COCO class IDs that count as "holdable" objects.
# YOLOv8n uses COCO labels — cell phone is class 67.
# Expand this list if you want to detect other objects too.
HOLDABLE_CLASSES = {
    67: "cell phone",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    73: "book",
    39: "bottle",
    41: "cup",
    76: "scissors",
    79: "toothbrush",
        # COCO grocery / kitchen items (15)
    40: "wine glass",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",

    # New tech items (headphones, wearables, accessories) (30)
    80: "headphones",
    81: "earbuds",
    82: "smartwatch",
    83: "tablet",
    84: "power bank",
    85: "charger",
    86: "usb cable",
    87: "bluetooth speaker",
    88: "webcam",
    89: "microphone",
    90: "smartphone stand",
    91: "smart glasses",
    92: "fitness tracker",
    93: "gaming controller",
    94: "external hard drive",
    95: "ssd",
    96: "usb flash drive",
    97: "smart plug",
    98: "smart bulb",
    99: "router",
    100: "smart home hub",
    101: "vr headset",
    102: "action camera",
    103: "drone",
    104: "stylus pen",
    105: "digital pen",
    106: "graphics tablet",
    107: "portable monitor",
    108: "keyboard cover",
    109: "laptop stand",

    # Additional grocery / supermarket goods (45)
    110: "milk carton",
    111: "yogurt cup",
    112: "cheese block",
    113: "butter stick",
    114: "eggs carton",
    115: "bread loaf",
    116: "bagel",
    117: "croissant",
    118: "cereal box",
    119: "pasta box",
    120: "rice bag",
    121: "flour bag",
    122: "sugar bag",
    123: "salt shaker",
    124: "pepper grinder",
    125: "olive oil bottle",
    126: "vinegar bottle",
    127: "ketchup bottle",
    128: "mustard bottle",
    129: "mayonnaise jar",
    130: "jam jar",
    131: "honey jar",
    132: "peanut butter jar",
    133: "nutella jar",
    134: "canned beans",
    135: "canned tuna",
    136: "canned soup",
    137: "soda can",
    138: "juice box",
    139: "water bottle",
    140: "coffee bag",
    141: "tea box",
    142: "chocolate bar",
    143: "candy bag",
    144: "chips bag",
    145: "pretzel bag",
    146: "popcorn bag",
    147: "ice cream tub",
    148: "frozen pizza box",
    149: "frozen vegetables bag",
    150: "tofu pack",
    151: "meat tray",
    152: "fish fillet pack",
    153: "deli meat pack",
    154: "baby food jar",
}


# --- 2. DOWNLOAD POSE MODEL IF MISSING ---
if not os.path.exists(POSE_MODEL_PATH):
    print("Downloading MediaPipe pose model...")
    urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)

# --- 3. INITIALIZE MODELS ---
yolo_model = YOLO(YOLO_MODEL_NAME)

base_options = mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
options = mp_vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = mp_vision.PoseLandmarker.create_from_options(options)


# --- 4. HELPER FUNCTIONS ---

def boxes_are_close(person_box, obj_box, proximity_px):
    """Check if object bounding box is within proximity_px of, or inside, person box."""
    px1, py1, px2, py2 = person_box
    ox1, oy1, ox2, oy2 = obj_box
    return not (
        ox1 > px2 + proximity_px or
        ox2 < px1 - proximity_px or
        oy1 > py2 + proximity_px or
        oy2 < py1 - proximity_px
    )


def box_is_inside(person_box, obj_box):
    """
    FIX: Check if the object box is fully or mostly inside the person box.
    This catches phones/remotes that are swallowed by the person bounding box
    and suppressed by NMS before they reach boxes_are_close().
    Returns True if >60% of the object area overlaps with the person box.
    """
    px1, py1, px2, py2 = person_box
    ox1, oy1, ox2, oy2 = obj_box

    ix1 = max(px1, ox1)
    iy1 = max(py1, oy1)
    ix2 = min(px2, ox2)
    iy2 = min(py2, oy2)

    if ix2 <= ix1 or iy2 <= iy1:
        return False

    intersection = (ix2 - ix1) * (iy2 - iy1)
    obj_area = max((ox2 - ox1) * (oy2 - oy1), 1)

    return (intersection / obj_area) > 0.60


def check_pocket(landmarks):
    """Detect if hands are very close to hip landmarks."""
    r_wrist, r_hip = landmarks[16], landmarks[24]
    l_wrist, l_hip = landmarks[15], landmarks[23]

    dist_r = math.sqrt((r_wrist.x - r_hip.x)**2 + (r_wrist.y - r_hip.y)**2)
    dist_l = math.sqrt((l_wrist.x - l_hip.x)**2 + (l_wrist.y - l_hip.y)**2)

    return dist_r < POCKET_THRESHOLD, dist_l < POCKET_THRESHOLD


def capture_and_save(frame):
    """Save a screenshot to disk and record it in the database."""
    os.makedirs("static/screenshots", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"screenshot_{timestamp}.png"
    filepath = f"static/screenshots/{filename}"

    cv2.imwrite(filepath, frame)

    with app.app_context():
        new_entry = Photo(filename=filename, filepath=filepath, upload_time=datetime.utcnow())
        db.session.add(new_entry)
        db.session.commit()

    print(f"[Screenshot] Saved {filename}")


# --- 5. MAIN LOOP ---

def generate_frames():
    global LAST_CAPTURE_TIME

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Live Stream Detection Running... Streaming to web.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int(time.time() * 1000)
            h, w = frame.shape[:2]
            proximity_px = w * PROXIMITY_RATIO

            # --- PHASE A: YOLO OBJECT DETECTION ---

            yolo_results = yolo_model(
                frame,
                conf=0.25,
                agnostic_nms=True,
                verbose=False
            )[0]

            person_boxes = []
            object_boxes = []

            for box in yolo_results.boxes:
                cls_id = int(box.cls[0])
                label = yolo_model.names[cls_id]
                coords = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, coords)

                if label == "person":
                    color = (0, 255, 0)
                    person_boxes.append(coords)
                elif cls_id in HOLDABLE_CLASSES:
                    # FIX: Only track holdable objects, ignoring furniture/vehicles etc.
                    color = (0, 165, 255)
                    object_boxes.append(coords)
                else:
                    color = (180, 180, 180)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # FIX: Check both proximity AND containment.
            # A held phone is usually inside the person box, not just near it.
            holding_detected = False
            for person_box in person_boxes:
                for obj_box in object_boxes:
                    if boxes_are_close(person_box, obj_box, proximity_px) or \
                       box_is_inside(person_box, obj_box):
                        holding_detected = True
                        break
                if holding_detected:
                    break

            # FIX: Fallback — if NMS removed the object box entirely, re-run
            # YOLO on just the person crop at lower confidence to find it.
            if not holding_detected and person_boxes:
                for person_box in person_boxes:
                    px1, py1, px2, py2 = map(int, person_box)
                    pad = 10
                    crop = frame[
                        max(0, py1 + pad): min(h, py2 - pad),
                        max(0, px1 + pad): min(w, px2 - pad)
                    ]
                    if crop.size == 0:
                        continue

                    crop_results = yolo_model(crop, conf=0.20, verbose=False)[0]
                    for box in crop_results.boxes:
                        cls_id = int(box.cls[0])
                        if cls_id in HOLDABLE_CLASSES:
                            holding_detected = True
                            # Translate crop-relative coords back to full frame
                            cx1, cy1, cx2, cy2 = map(int, box.xyxy[0].tolist())
                            fx1, fy1 = px1 + pad + cx1, py1 + pad + cy1
                            fx2, fy2 = px1 + pad + cx2, py1 + pad + cy2
                            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 165, 255), 2)
                            cv2.putText(frame, HOLDABLE_CLASSES[cls_id],
                                        (fx1, fy1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                            break
                    if holding_detected:
                        break

            if holding_detected:
                cv2.putText(
                    frame, "HOLDING OBJECT",
                    (20, 80),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1, (0, 255, 255), 2
                )

            # --- PHASE B: MEDIAPIPE POSE DETECTION ---
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            pose_result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if pose_result.pose_landmarks:
                for pose_landmarks in pose_result.pose_landmarks:
                    right_in, left_in = check_pocket(pose_landmarks)

                    if (right_in and left_in) and holding_detected:
                        current_time = time.time()
                        if current_time - LAST_CAPTURE_TIME >= COOLDOWN_SECONDS:
                            LAST_CAPTURE_TIME = current_time
                            threading.Thread(
                                target=capture_and_save,
                                args=(frame.copy(),),
                                daemon=True
                            ).start()

            # --- PHASE C: STREAM OUTPUT ---
            ret_enc, buffer = cv2.imencode('.jpg', frame)
            if ret_enc:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    finally:
        cap.release()
        print("Camera released.")