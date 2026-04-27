import cv2
import numpy as np
import time
import torch
import os
from picamera2 import Picamera2

# ====================== CONFIG ======================
# Choose your model here:
MODEL_CONFIG = {
    "name": "dpt_swin2_tiny_256",           # change to "dpt_levit_224" if you want the fastest
    "hub_entry": "DPT_SwinV2_T_256",
    "transform": "swin256_transform",
    "file": "dpt_swin2_tiny_256.pt"
}

# For LeViT instead, use this:
# MODEL_CONFIG = {
#     "name": "dpt_levit_224",
#     "hub_entry": "DPT_LeViT_224",
#     "transform": "levit_transform",
#     "file": "dpt_levit_224.pt"
# }

PREVIEW_SIZE = (640, 480)        # reduce to (480, 360) if still slow

if not os.path.exists(MODEL_CONFIG["file"]):
    print(f"ERROR: Model file '{MODEL_CONFIG['file']}' not found!")
    print(f"Download it from: https://github.com/isl-org/MiDaS/releases/download/v3_1/{MODEL_CONFIG['file']}")
    exit(1)

print(f"Loading {MODEL_CONFIG['name']} model...")

midas = torch.hub.load("isl-org/MiDaS", MODEL_CONFIG["hub_entry"], pretrained=False)
state_dict = torch.load(MODEL_CONFIG["file"], map_location=torch.device('cpu'), weights_only=True)
midas.load_state_dict(state_dict)

device = torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("isl-org/MiDaS", "transforms")
transform = getattr(midas_transforms, MODEL_CONFIG["transform"])   # this line handles the difference

print(f"{MODEL_CONFIG['name']} loaded successfully on {device}")


prev_gray = None
flow_params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

alert_history = []
HISTORY_LENGTH = 8

def estimate_depth(rgb_image):
    input_tensor = transform(rgb_image)
    if input_tensor.dim() == 5:
        input_tensor = input_tensor.squeeze(1)
    elif input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)
    
    input_batch = input_tensor.to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=rgb_image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    depth = prediction.cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth


def detect_motion_type(current_gray, prev_gray, depth_map):
    if prev_gray is None:
        return "WARMING UP", 0.0, (200, 200, 200), "", 0.0, 0.0

    flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, **flow_params)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    h, w = depth_map.shape
    roi_h = slice(h//4, 3*h//4)
    roi_w = slice(w//5, 4*w//5)

    center_depth = depth_map[roi_h, roi_w]
    center_flow = magnitude[roi_h, roi_w]

    avg_closeness = float(center_depth.mean())
    avg_flow = float(center_flow.mean())

    relative_speed = avg_flow * (avg_closeness ** 1.8)

    # === Updated Logic based on your feedback ===
    if avg_closeness > 0.55 and relative_speed > 6.0:          # Strong closeness + decent motion → FAST
        status = "FAST APPROACHING OBJECT!"
        color = (0, 0, 255)      # Red
        alert = "Would trigger sound"
    elif avg_closeness > 0.45 or (avg_closeness > 0.35 and relative_speed > 3.0):
        status = "MOVING TOWARD OBSTACLE"
        color = (0, 165, 255)    # Orange
        alert = "Would trigger sound"
    else:
        status = "Normal / Slow Motion"
        color = (0, 255, 0)
        alert = ""

    return status, relative_speed, color, alert, avg_closeness, avg_flow


def main():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": PREVIEW_SIZE, "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    print(f"Camera started at {PREVIEW_SIZE}")

    print(f"\nMoving Targets Test Running with {MODEL_CONFIG['name']}")
    print("Press 'q' to quit\n")

    global prev_gray
    last_time = time.time()
    frame_count = 0

    while True:
        array = picam2.capture_array("main")
        frame = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

        frame_count += 1
        current_time = time.time()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        depth_map = estimate_depth(rgb_frame)

        status, rel_speed, text_color, alert, closeness, flow = detect_motion_type(gray, prev_gray, depth_map)
        prev_gray = gray.copy()

        # Alert smoothing + visualization code (same as your previous version)
        alert_history.append((status, text_color, alert))
        if len(alert_history) > HISTORY_LENGTH:
            alert_history.pop(0)

        display_status, display_color, display_alert = max(
            alert_history, 
            key=lambda x: 2 if "FAST" in x[0] else 1 if "OBSTACLE" in x[0] else 0
        )

        depth_vis = (depth_map * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 0.6, depth_color, 0.4, 0)

        cv2.putText(overlay, display_status, (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.1, display_color, 3)
        if display_alert:
            cv2.putText(overlay, display_alert, (30, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.85, display_color, 2)
        cv2.putText(overlay, f"Closeness: {closeness:.2f} | Speed: {rel_speed:.1f}", 
                    (30, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if current_time - last_time > 1.0:
            fps = frame_count / (current_time - last_time)
            cv2.putText(overlay, f"FPS: {fps:.1f}", (overlay.shape[1]-160, overlay.shape[0]-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            last_time = current_time
            frame_count = 0

        cv2.imshow(f"Moving Targets - {MODEL_CONFIG['name']}", overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
