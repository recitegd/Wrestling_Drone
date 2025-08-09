import pyrealsense2 as rs
import numpy as np
import cv2
import threading
import time
from typing import Optional, Tuple, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealSenseCamera:
    def __init__(self, device_id: Optional[str] = None):
        self._pipeline = None
        self._config = None
        self._device_id = device_id
        self._is_streaming = False
        self._lock = threading.Lock()
        
        self._latest_frame = None
        self._frame_timestamp = None
        self._width = 640
        self._height = 480
        self._fps = 30
        
        # MediaPipe compatibility
        self._flip_horizontal = False
        self._convert_bgr_to_rgb = True
        
        logger.info("RealSense camera manager initialized")
    
    def configure_stream(self, resolution: Tuple[int, int] = (640, 480),fps: int = 30) -> None:
        if self._is_streaming:
            raise RuntimeError("Cannot configure stream while camera is streaming")
        
        self._width, self._height = resolution
        self._fps = fps
        
        logger.info(f"Configured stream - Color: {resolution}@{fps}fps")
    
    def set_mediapipe_options(self, flip_horizontal: bool = False, convert_bgr_to_rgb: bool = True) -> None:
        self._flip_horizontal = flip_horizontal
        self._convert_bgr_to_rgb = convert_bgr_to_rgb
        logger.info(f"MediaPipe options - Flip: {flip_horizontal}, BGR->RGB: {convert_bgr_to_rgb}")
    
    def _initialize_pipeline(self) -> bool:
        try:
            self._pipeline = rs.pipeline()
            self._config = rs.config()
            
            # Configure device if specified
            if self._device_id:
                self._config.enable_device(self._device_id)
            
            # Configure color stream only
            self._config.enable_stream(rs.stream.color, 
                                     self._width, self._height, 
                                     rs.format.bgr8, self._fps)
            
            # Test configuration
            profile = self._pipeline.start(self._config)
            
            logger.info("Pipeline initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            self._cleanup_pipeline()
            return False
    
    def _cleanup_pipeline(self) -> None:
        """Clean up pipeline resources."""
        if self._pipeline:
            try:
                self._pipeline.stop()
            except Exception as e:
                logger.warning(f"Error stopping pipeline: {e}")
            finally:
                self._pipeline = None
                self._config = None
    
    def start(self) -> bool:
        with self._lock:
            if self._is_streaming:
                logger.warning("Camera is already streaming")
                return True
            
            if not self._initialize_pipeline():
                return False
            
            self._is_streaming = True
            logger.info("Camera streaming started")
            return True
    
    def stop(self) -> None:
        with self._lock:
            if not self._is_streaming:
                logger.warning("Camera is not streaming")
                return
            
            self._is_streaming = False
            self._cleanup_pipeline()
            self._latest_frame = None
            self._frame_timestamp = None
            
            logger.info("Camera streaming stopped")
    
    def get_frame(self, timeout_ms: int = 5000) -> Optional[np.ndarray]:
        """
        Returns color frame as numpy array, or None if failed
        """
        if not self._is_streaming:
            logger.warning("Camera is not streaming")
            return None
        
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms)
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                logger.warning("Failed to get color frame")
                return None
            
            color_image = np.asanyarray(color_frame.get_data())
            
            # Apply MediaPipe-specific transformations
            if self._convert_bgr_to_rgb:
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            if self._flip_horizontal:
                color_image = cv2.flip(color_image, 1)
            
            self._latest_frame = color_image
            self._frame_timestamp = time.time()
            
            return color_image.copy()
            
        except Exception as e:
            logger.error(f"Error getting frame: {e}")
            return None
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the most recently cached frame without waiting."""
        return self._latest_frame.copy() if self._latest_frame is not None else None
    
    def get_frame_timestamp(self) -> Optional[float]:
        return self._frame_timestamp
    
    def get_intrinsics(self) -> Optional[Dict[str, Any]]:
        """Get camera intrinsic parameters."""
        if not self._is_streaming:
            return None
        
        try:
            profile = self._pipeline.get_active_profile()
            color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
            color_intrinsics = color_profile.get_intrinsics()
            
            return {
                'width': color_intrinsics.width,
                'height': color_intrinsics.height,
                'fx': color_intrinsics.fx,
                'fy': color_intrinsics.fy,
                'ppx': color_intrinsics.ppx,
                'ppy': color_intrinsics.ppy,
                'coeffs': color_intrinsics.coeffs
            }
        except Exception as e:
            logger.error(f"Error getting intrinsics: {e}")
            return None
    
    def __enter__(self):
        """Context manager entry."""
        if not self.start():
            raise RuntimeError("Failed to start camera")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        if self._is_streaming:
            self.stop()


def create_mediapipe_camera(device_id: Optional[str] = None, resolution: Tuple[int, int] = (640, 480),fps: int = 30) -> RealSenseCamera:
    """
    Create a RealSense camera optimized for MediaPipe.
    """
    camera = RealSenseCamera(device_id)
    camera.configure_stream(resolution, fps)
    camera.set_mediapipe_options(flip_horizontal=False, convert_bgr_to_rgb=True)
    
    return camera