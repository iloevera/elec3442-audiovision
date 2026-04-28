from __future__ import annotations

from dataclasses import dataclass
import time
import cv2
import numpy as np
#from picamera2 import Picamera2


@dataclass(frozen=True)
class TripCameraFrame:
    image: np.ndarray
    timestamp_s: float


class PiTripCamera:
    #def __init__(self, width: int = 640, height: int = 480)
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self._cap: cv2.VideoCapture | None = None
        #self._cam: Picamera2 | None = None

    def start(self) -> None:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self.camera_index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap = cap
        '''cam = Picamera2()
        config = cam.create_preview_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"}
        )
        cam.configure(config)
        cam.start()
        time.sleep(0.2)
        self._cam = cam'''
        
    def read(self) -> TripCameraFrame | None:
        if self._cap is None:
            raise RuntimeError("PiTripCamera.start() must be called first")
        ok, frame = self._cap.read()
        if not ok:
            return None
        return TripCameraFrame(image=frame, timestamp_s=time.monotonic())
        '''
        if self._cam is None:
            raise RuntimeError("PiTripCamera.start() must be called first")
        frame = self._cam.capture_array()
        return TripCameraFrame(image=frame, timestamp_s=time.monotonic())
        '''

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        #if self._cam is not None:
        #    self._cam.stop()
        #    self._cam = None

    def __enter__(self) -> "PiTripCamera":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()