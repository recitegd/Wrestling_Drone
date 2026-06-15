from ultralytics import YOLO
import cv2
import camera_handler
import threading
import media_pipe_handler
import os

model = YOLO("yolov8n.pt")
running = True
frame_results = []
frame_lock = threading.Lock()
CONFIDENCE_THRESHOLD = 0.5
MAX_WRESTLERS = 2
WINDOW_NAME = "Wrestling Coach"
fullscreen = False
button_rects = {}

def camera_stream_thread():
    global frame_results, running
    frame = None
    setup_window()
    with camera_handler.create_mediapipe_camera() as camera:
        while running:
            frame = camera.get_frame()
            if frame is not None:
                people = detect_people(frame)
                with frame_lock:
                    frame_results = people
                media_pipe_handler.process_wrestler_frames(people)

                display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                media_pipe_handler.draw_pose_landmarks(display_frame, people)
                draw_detections(display_frame, people)

                cv2.imshow(WINDOW_NAME, display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                end_program()

def setup_window():
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 720)

def toggle_fullscreen():
    global fullscreen
    fullscreen = not fullscreen
    if fullscreen:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 960, 720)

def detect_people(frame):
    people = []
    height, width = frame.shape[:2]
    yolo_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # YOLO tracking gives us stable IDs when possible; sorted fallback labels keep prompts deterministic.
    results = model.track(yolo_frame, classes=[0], persist=True, verbose=False)
    boxes = results[0].boxes if results and results[0].boxes is not None else []
    for index, box in enumerate(boxes):
        cls = int(box.cls[0])
        if cls != 0:
            continue

        confidence = float(box.conf[0])
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0]]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        if x2 <= x1 or y2 <= y1:
            continue

        track_id = None
        if box.id is not None:
            track_id = int(box.id[0])

        people.append({
            "id": track_id if track_id is not None else index + 1,
            "label": f"Wrestler {track_id if track_id is not None else index + 1}",
            "frame": frame[y1:y2, x1:x2],
            "box": (x1, y1, x2, y2),
            "confidence": confidence,
        })

    people.sort(key=lambda person: (person["box"][0], person["box"][1]))
    for index, person in enumerate(people[:MAX_WRESTLERS], start=1):
        if person["id"] is None:
            person["id"] = index
        person["label"] = f"Wrestler {index}"

    return people[:MAX_WRESTLERS]

def draw_detections(display_frame, people):
    for person in people:
        x1, y1, x2, y2 = person["box"]
        label = f'{person["label"]} {person["confidence"]:.2f}'
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 220, 120), 2)
        cv2.putText(display_frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 120), 2)

def end_program():
    global running
    running = False
    media_pipe_handler.running = False
    try:
        import input_output
        input_output.listen_and_speak = False
    except ImportError:
        pass

    print("Stopping YOLO...")
    cv2.destroyAllWindows()
    os._exit(0)
