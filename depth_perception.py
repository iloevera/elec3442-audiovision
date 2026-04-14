import cv2
from picamera2 import Picamera2
import time

camera0 = Picamera2(cam_num=0)
camera1 = Picamera2(cam_num=1)

config0 = camera0.create_preview_configuration(main={"size": (640, 480)})
config1 = camera1.create_preview_configuration(main={"size": (640, 480)})

camera0.configure(config0)
camera1.configure(config1)

camera0.start()
camera1.start()
time.sleep(1)
