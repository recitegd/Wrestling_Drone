from picamera2 import Picamera2
import cv2
import numpy as np

class Camera:

    def __init__(self):
        self._cam = Picamera2()
        self._is_streaming = False
        self._width = 640
        self._height = 480
        self._fps = 30

    def configure_stream(self, resolution=(640, 480), fps=30):
        self._width, self._height = resolution
        self._fps = fps

        config = self._cam.create_video_configuration(
            main={"size": resolution}
        )
        self._cam.configure(config)

    def start(self):
        self._cam.start()
        self._is_streaming = True
        return True
    
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def stop(self):
        self._cam.stop()
        self._is_streaming = False

    def get_frame(self):
        if not self._is_streaming:
            return None

        frame = self._cam.capture_array()

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        return frame