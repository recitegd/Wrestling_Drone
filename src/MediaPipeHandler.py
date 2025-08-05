import cv2
import mediapipe as mp
import Camera

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
running = True

with Camera.create_mediapipe_camera() as camera:
    while running:
        frame = camera.get_frame()
        if frame is not None:
            processedFrame = pose.process(frame)
            displayFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if processedFrame.pose_landmarks:
                mpDraw.draw_landmarks(displayFrame, processedFrame.pose_landmarks, mpPose.POSE_CONNECTIONS)

            cv2.imshow("Camera", displayFrame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
        else:
            print("failed")

cv2.destroyAllWindows()