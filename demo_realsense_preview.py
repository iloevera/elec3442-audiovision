from __future__ import annotations

from contextlib import suppress

import cv2
import numpy as np

from realsense_driver import D435iDriver, FrameBundle, IMUSample


WINDOW_NAME = "RealSense D435i"
HEADER_HEIGHT = 72
FOOTER_HEIGHT = 164
WINDOW_SIZE = (1440, 940)


def run_preview(window_name: str = WINDOW_NAME) -> None:
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, *WINDOW_SIZE)

    with D435iDriver() as driver:
        try:
            while True:
                bundle = driver.wait_for_bundle(timeout_s=1.0)
                if bundle is None:
                    if driver.last_error is not None:
                        raise RuntimeError("Capture thread stopped after an error") from driver.last_error
                    continue

                preview = compose_preview_frame(bundle=bundle, imu_enabled=driver.imu_enabled_runtime)
                cv2.imshow(window_name, preview)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
        finally:
            with suppress(Exception):
                cv2.destroyWindow(window_name)
            cv2.destroyAllWindows()


def compose_preview_frame(bundle: FrameBundle, imu_enabled: bool) -> np.ndarray:
    depth_color = cv2.applyColorMap(
        cv2.convertScaleAbs(bundle.depth.image, alpha=0.03),
        cv2.COLORMAP_JET,
    )
    color_image = bundle.color.image

    if color_image.shape[:2] != depth_color.shape[:2]:
        depth_color = cv2.resize(depth_color, (color_image.shape[1], color_image.shape[0]))

    preview = np.hstack((color_image, depth_color))
    preview = cv2.copyMakeBorder(
        preview,
        top=HEADER_HEIGHT,
        bottom=FOOTER_HEIGHT,
        left=0,
        right=0,
        borderType=cv2.BORDER_CONSTANT,
        value=(18, 18, 18),
    )

    width = preview.shape[1]
    header = preview[:HEADER_HEIGHT]
    footer = preview[-FOOTER_HEIGHT:]
    header[:] = vertical_gradient(header.shape[0], width, (36, 22, 12), (12, 12, 12))
    footer[:] = vertical_gradient(footer.shape[0], width, (14, 14, 14), (28, 28, 28))

    cv2.line(preview, (0, HEADER_HEIGHT), (width, HEADER_HEIGHT), (60, 98, 222), 2, cv2.LINE_AA)
    cv2.line(
        preview,
        (width // 2, HEADER_HEIGHT),
        (width // 2, preview.shape[0] - FOOTER_HEIGHT),
        (42, 42, 42),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        preview,
        "Intel RealSense D435i Live Preview",
        (18, 30),
        cv2.FONT_HERSHEY_DUPLEX,
        0.9,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        f"Depth {bundle.depth.image.shape[1]}x{bundle.depth.image.shape[0]}   Color {bundle.color.image.shape[1]}x{bundle.color.image.shape[0]}",
        (18, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (188, 198, 212),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        preview,
        "COLOR",
        (18, 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        "DEPTH",
        (width // 2 + 18, 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    draw_status_chip(
        preview,
        label=("IMU ONLINE" if imu_enabled else "IMU OFF"),
        origin=(width - 194, 18),
        size=(176, 34),
        color=((48, 166, 98) if imu_enabled else (88, 88, 88)),
    )

    accel_lines = format_imu_lines("ACCEL", bundle.latest_accel, "m/s^2")
    gyro_lines = format_imu_lines("GYRO", bundle.latest_gyro, "rad/s")
    footer_lines = [
        f"Bundle IMU samples: {len(bundle.imu_samples)}",
        f"Depth frame #{bundle.depth.frame_number} at {bundle.depth.timestamp_ms:8.2f} ms",
        f"Color frame #{bundle.color.frame_number} at {bundle.color.timestamp_ms:8.2f} ms",
        *accel_lines,
        *gyro_lines,
        "Press Q or Esc to exit",
    ]

    line_y = preview.shape[0] - FOOTER_HEIGHT + 28
    for line in footer_lines:
        cv2.putText(
            preview,
            line,
            (18, line_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (236, 236, 236),
            1,
            cv2.LINE_AA,
        )
        line_y += 18

    return preview


def vertical_gradient(
    height: int,
    width: int,
    top_bgr: tuple[int, int, int],
    bottom_bgr: tuple[int, int, int],
) -> np.ndarray:
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    for channel, (top_value, bottom_value) in enumerate(zip(top_bgr, bottom_bgr)):
        gradient[:, :, channel] = np.linspace(top_value, bottom_value, height, dtype=np.uint8)[:, None]
    return gradient


def draw_status_chip(
    image: np.ndarray,
    *,
    label: str,
    origin: tuple[int, int],
    size: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    x, y = origin
    width, height = size
    cv2.rectangle(image, (x, y), (x + width, y + height), color, thickness=-1)
    cv2.rectangle(image, (x, y), (x + width, y + height), (240, 240, 240), thickness=1)
    cv2.putText(
        image,
        label,
        (x + 14, y + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def format_imu_lines(label: str, sample: IMUSample | None, units: str) -> list[str]:
    if sample is None:
        return [f"{label}: waiting for data"]

    magnitude = float(np.linalg.norm(sample.xyz))
    return [
        f"{label}: x={sample.xyz[0]: .3f}  y={sample.xyz[1]: .3f}  z={sample.xyz[2]: .3f} {units}",
        f"{label}: |v|={magnitude: .3f} {units}  frame #{sample.frame_number}  ts={sample.timestamp_ms:8.2f} ms",
    ]


if __name__ == "__main__":
    run_preview()