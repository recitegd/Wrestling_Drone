import cv2
import numpy as np
import mediapipe as mp
import json
import os
import math
import threading
import sys
from collections import deque
from datetime import datetime

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
PoseLandmark = mp.solutions.pose.PoseLandmark

joint_angles = {}
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "joint_data.json")
with open(file_path, "r") as f:
    joint_angles = json.load(f)["joint_angles"]


with open(file_path, "r") as f:
    joint_positions = json.load(f)["joint_positions"]

MAX_CACHE_LEN = 5
VISIBILITY_THRESHOLD = 0.85
MAX_FRAME_AGE_SECONDS = 2
MAX_WRESTLERS = 2

cache_lock = threading.Lock()
running = True
frame_results = {}
poses = {}
wrestler_caches = {}

def create_angle_cache():
    return [{"name": angle_name, "frames": deque([])} for angle_name in joint_angles]

def create_position_cache():
    return [{"name": pos, "frames": deque([])} for pos in joint_positions]

def get_wrestler_cache(wrestler_id):
    if wrestler_id not in wrestler_caches:
        wrestler_caches[wrestler_id] = {
            "label": f"Wrestler {wrestler_id}",
            "angle_cache": create_angle_cache(),
            "position_cache": create_position_cache(),
            "last_seen": None,
            "box": None,
            "confidence": None,
        }
    return wrestler_caches[wrestler_id]

def get_pose(wrestler_id):
    if wrestler_id not in poses:
        poses[wrestler_id] = mp_pose.Pose()
    return poses[wrestler_id]

def is_joint_angle_visible(joint, mp_result):
    if mp_result.pose_landmarks is None:
        return False
    landmark = mp_result.pose_landmarks.landmark
    points = joint_angles[joint["name"]]
    if landmark[PoseLandmark[points[0]]].visibility > VISIBILITY_THRESHOLD and landmark[PoseLandmark[points[1]]].visibility > VISIBILITY_THRESHOLD and landmark[PoseLandmark[points[2]]].visibility > VISIBILITY_THRESHOLD:
        return True
    return False

def get_joint_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    
    dot_product = np.dot(ba, bc)
    mag_ba = np.linalg.norm(ba)
    mag_bc = np.linalg.norm(bc)
    
    cos_angle = dot_product / (mag_ba * mag_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    return round(angle_deg, 3)

def record_pose_result(wrestler_id, result, label=None, box=None, confidence=None):
    if result.pose_landmarks is None:
        return

    now = datetime.now()
    cache = get_wrestler_cache(wrestler_id)
    cache["label"] = label or cache["label"]
    cache["last_seen"] = now
    cache["box"] = box
    cache["confidence"] = confidence
    frame_results[wrestler_id] = result

    landmark = result.pose_landmarks.landmark
    for joint in cache["angle_cache"]:
        if not is_joint_angle_visible(joint, result):
            continue
        if len(joint["frames"]) >= MAX_CACHE_LEN:
            joint["frames"].popleft()

        points = joint_angles[joint["name"]]
        ja = landmark[PoseLandmark[points[0]]]
        jb = landmark[PoseLandmark[points[1]]]
        jc = landmark[PoseLandmark[points[2]]]
        a = (ja.x, ja.y, ja.z)
        b = (jb.x, jb.y, jb.z)
        c = (jc.x, jc.y, jc.z)
        angle = get_joint_angle(a, b, c)
        if angle is not None and not math.isnan(angle):
            joint["frames"].append({"angle": round(angle), "timestamp": now})

    for joint_obj in cache["position_cache"]:
        joint = landmark[PoseLandmark[joint_obj["name"]]]
        if joint.visibility < VISIBILITY_THRESHOLD:
            continue
        if len(joint_obj["frames"]) >= MAX_CACHE_LEN:
            joint_obj["frames"].popleft()
        joint_obj["frames"].append({
            "position": (round(joint.x, 3), round(joint.y, 3), round(joint.z, 3)),
            "timestamp": now,
        })

def process_wrestler_frames(wrestler_frames):
    with cache_lock:
        for wrestler in wrestler_frames[:MAX_WRESTLERS]:
            wrestler_id = wrestler["id"]
            crop = wrestler["frame"]
            if crop is None or crop.size == 0:
                continue

            result = get_pose(wrestler_id).process(crop)
            record_pose_result(
                wrestler_id,
                result,
                label=wrestler.get("label"),
                box=wrestler.get("box"),
                confidence=wrestler.get("confidence"),
            )

def draw_pose_landmarks(display_frame, wrestler_frames):
    with cache_lock:
        for wrestler in wrestler_frames:
            result = frame_results.get(wrestler["id"])
            if result is None or result.pose_landmarks is None:
                continue

            x1, y1, x2, y2 = wrestler["box"]
            display_crop = display_frame[y1:y2, x1:x2]
            if display_crop.size == 0:
                continue

            mp_draw.draw_landmarks(display_crop, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

def extract_angles(angle_cache):
    joints = {}

    for joint in angle_cache:
        if not joint["frames"]: continue
        control_point = joint["frames"][-1]
        angles_to_average = []
        #reversed to traverse from youngest to oldest, allowing the "break" statement
        for angle_info in reversed(joint["frames"]):
            if (angle_info["timestamp"] - control_point["timestamp"]).total_seconds() > MAX_FRAME_AGE_SECONDS: break
            angles_to_average.append(angle_info["angle"])
        
        size = len(angles_to_average)
        if size == 0: continue
        total = 0
        for angle in angles_to_average: total += angle
        joints[joint["name"]] = round(total / size, 3)

    return joints

def extract_positions(position_cache):
    joints = {}

    for joint in position_cache:
        if not joint["frames"]: continue

        control_point = joint["frames"][-1]
        pos_to_average = []

        for pos_info in reversed(joint["frames"]):
            if (control_point["timestamp"] - pos_info["timestamp"]).total_seconds() > MAX_FRAME_AGE_SECONDS: break
            pos_to_average.append(pos_info["position"])  # (x, y, z)

        size = len(pos_to_average)
        if size == 0:continue
        total_x = total_y = total_z = 0
        for x, y, z in pos_to_average:
            total_x += x
            total_y += y
            total_z += z
        joints[joint["name"]] = (round(total_x / size, 3), round(total_y / size, 3), round(total_z / size, 3))

    return joints

def construct_prompt(wrestlers):
    parts = []
    parts.append("\n\nCurrent mat vision data:\n")
    parts.append("Each wrestler entry is separate. Joint angles are degrees. Positions are normalized crop coordinates (x, y, z), so compare posture within each wrestler more than exact mat distance.\n")

    for wrestler in wrestlers:
        parts.append(f"\n{wrestler['label']}:\n")
        if wrestler.get("confidence") is not None:
            parts.append(f"detector_confidence: {round(float(wrestler['confidence']), 3)}\n")

        parts.append("joint_angles:\n")
        for joint, angle in wrestler["angles"].items():
            parts.append(f"- {joint}: {angle}\n")

        parts.append("joint_positions:\n")
        for joint, position in wrestler["positions"].items():
            parts.append(f"- {joint}: {position}\n")

    prompt = "".join(parts)
    return prompt

def end_program():
    global running
    running = False
    print("Stopping program...")
    cv2.destroyAllWindows()
    for pose_instance in poses.values():
        pose_instance.close()
    sys.exit()

class MediaPipeHandler:
    def create_request(self):
        with cache_lock:
            wrestlers = []
            now = datetime.now()
            for wrestler_id, cache in sorted(wrestler_caches.items()):
                if cache["last_seen"] is None:
                    continue
                if (now - cache["last_seen"]).total_seconds() > MAX_FRAME_AGE_SECONDS:
                    continue

                angles = extract_angles(cache["angle_cache"])
                positions = extract_positions(cache["position_cache"])
                if not angles and not positions:
                    continue

                wrestlers.append({
                    "id": wrestler_id,
                    "label": cache["label"],
                    "confidence": cache["confidence"],
                    "angles": angles,
                    "positions": positions,
                })

            if wrestlers:
                return construct_prompt(wrestlers)
            return False
