import cv2
import numpy as np
import mediapipe as mp
import Camera
import json
import time
import os
import math
import threading
import sys
from collections import deque
from datetime import datetime

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils
PoseLandmark = mp.solutions.pose.PoseLandmark

joint_angles = {}
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "joint_data.json")
with open(file_path, "r") as f:
    joint_angles = json.load(f)["joint_angles"]


angle_cache = []
position_cache = []
timestamp = datetime.now()
for angle_name, joints_needed in joint_angles.items():
    joint = {"name": angle_name, "frames": deque([])}
    angle_cache.append(joint)
with open(file_path, "r") as f:
    joint_position = json.load(f)["joint_positions"]
    for pos in joint_position:
        position_cache.append({"name": pos, "frames": deque([])})

MAX_CACHE_LEN = 5
VISIBILITY_THRESHOLD = 0.85
MAX_FRAME_AGE_SECONDS = 2

def is_joint_angle_visible(joint, mp_result):
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
    return angle_deg


def extract_angles():
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
        joints[joint["name"]] = total / size

    return joints

def extract_positions():
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
        joints[joint["name"]] = (total_x / size, total_y / size, total_z / size)

    return joints



def construct_prompt(angles, positions):
    #also add further instruction like: what am i doing wrong here, give detailed, give quick, etc.
    #this first one should be given when first talking to the ai
    parts = ["Follow these instructions for every prompt given to you. You are a master folkstyle wrestling coach and you need to give instructions and tips to positions you see based on joint angles (degrees) given to you and questions that you may be asked. You also may need to answer questions.\n"]
    parts.append("What am I doing wrong here?\n")
    parts.append("Keep this response short.\n")
    parts.append("These are the angles for the following joints:\n")

    for joint, angle in angles.items():
        parts.append(f"{joint}: {angle}\n")

    parts.append("These are the relative positions for the following joints:\n")
    print(positions)
    for joint, position in positions.items():
        parts.append(f"{joint}: {position}\n")

    prompt = "".join(parts)
    return prompt

frame_lock = threading.Lock()
frame_updated = threading.Event()
running = True
frame_results = None

def end_program():
    global running
    running = False
    print("Stopping program...")
    cv2.destroyAllWindows()
    sys.exit()

def watch_feed_thread():
    global frame_results, running
    while running:
        frame_updated.wait()
        with frame_lock:
            if frame_results.pose_landmarks is None: continue
            landmark = frame_results.pose_landmarks.landmark
            for joint in angle_cache:
                if not is_joint_angle_visible(joint, frame_results): continue
                if len(joint["frames"]) >= MAX_CACHE_LEN: joint["frames"].popleft()

                points = joint_angles[joint["name"]]
                ja = landmark[PoseLandmark[points[0]]]
                jb = landmark[PoseLandmark[points[1]]]
                jc = landmark[PoseLandmark[points[2]]]
                a = (ja.x, ja.y, ja.z)
                b = (jb.x, jb.y, jb.z)
                c = (jc.x, jc.y, jc.z)
                angle = get_joint_angle(a, b, c)
                if angle is not None and not math.isnan(angle):
                    joint["frames"].append({"angle": angle, "timestamp": datetime.now()})
            
            for joint_obj in position_cache:
                joint = landmark[PoseLandmark[joint_obj["name"]]]
                if joint.visibility < VISIBILITY_THRESHOLD: continue
                if len(joint_obj["frames"]) >= MAX_CACHE_LEN: joint_obj["frames"].popleft()
                joint_obj["frames"].append({"position": (joint.x, joint.y, joint.z), "timestamp": datetime.now()})
            frame_updated.clear()


def camera_stream_thread():
    global frame_results, running
    frame = None
    previous_frame = None
    with Camera.create_mediapipe_camera() as camera:
        while running:
            previous_frame = frame
            frame = camera.get_frame()
            if frame is not None:
                with frame_lock:
                    frame_results = pose.process(frame)
                if(not np.array_equal(previous_frame, frame)): frame_updated.set()

                display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                if frame_results.pose_landmarks:
                    mp_draw.draw_landmarks(display_frame, frame_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                cv2.imshow("Camera", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                end_program()

def send_requests_thread():
    global frame_results, running
    time.sleep(5)
    while running:
        time.sleep(5)
        with frame_lock:
            if frame_results and frame_results.pose_landmarks:
                print(construct_prompt(extract_angles(), extract_positions()))


threading.Thread(target=camera_stream_thread, daemon=True).start()
threading.Thread(target=send_requests_thread, daemon=True).start()
threading.Thread(target=watch_feed_thread, daemon=True).start()

try:
    while True:
        time.sleep(0.5)
except KeyboardInterrupt:
    end_program()