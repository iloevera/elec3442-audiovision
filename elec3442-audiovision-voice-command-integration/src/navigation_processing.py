from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from .realsense_driver import CameraIntrinsics, FrameBundle


@dataclass(frozen=True)
class NavigationCellState:
    row: int
    col: int
    sample_count: int
    obstacle_fraction: float
    percentile_depth_m: Optional[float]
    approach_speed_mps: float
    ttc_s: Optional[float]
    risk_score: float


@dataclass(frozen=True)
class NavigationColumnState:
    col: int
    azimuth_deg: float
    sample_count: int
    risk_score: float
    percentile_depth_m: Optional[float]
    ttc_s: Optional[float]
    pitch_hz: float
    pulse_hz: float
    volume: float


@dataclass(frozen=True)
class GroundPlaneEstimate:
    normal: np.ndarray
    offset: float
    inlier_count: int
    mean_residual_m: float


@dataclass(frozen=True)
class NavigationFrameAnalysis:
    timestamp_s: float
    depth_m: np.ndarray
    valid_mask: np.ndarray
    ground_mask: np.ndarray
    obstacle_mask: np.ndarray
    height_above_ground_m: Optional[np.ndarray]
    gravity_unit: Optional[np.ndarray]
    ground_plane: Optional[GroundPlaneEstimate]
    cell_states: tuple[NavigationCellState, ...]
    column_states: tuple[NavigationColumnState, ...]
    depth_percentile: float
    percentile_depth_grid_m: np.ndarray
    ttc_grid_s: np.ndarray
    risk_grid: np.ndarray


@dataclass
class NavigationProcessorConfig:
    rows: int = 3
    cols: int = 5
    min_depth_m: float = 0.20
    max_depth_m: float = 4.00
    downsample_step: int = 2
    floor_candidate_fraction: float = 0.45
    ransac_iterations: int = 80
    ground_inlier_threshold_m: float = 0.035
    gravity_alignment_cos_min: float = 0.90
    max_plane_tilt_deg: float = 35.0
    min_plane_inliers: int = 300
    ground_plane_refit_interval_frames: int = 1
    floor_clearance_m: float = 0.04
    dropoff_clearance_m: float = 0.08
    min_obstacle_points_per_cell: int = 6
    depth_percentile: float = 5.0
    occupancy_denominator: int = 80
    ttc_min_speed_mps: float = 0.05
    ttc_min_depth_delta_m: float = 0.03
    ttc_min_dt_s: float = 0.01
    ttc_approach_ema_alpha: float = 0.40
    ttc_confirm_frames: int = 3
    ttc_clear_frames: int = 2
    ttc_max_speed_mps: float = 4.0
    ttc_horizon_s: float = 3.0
    proximity_near_m: float = 0.35
    proximity_far_m: float = 3.0
    pulse_rate_hz_min: float = 1.2
    pulse_rate_hz_max: float = 7.5
    pitch_hz_near: float = 1080.0
    pitch_hz_far: float = 260.0
    column_azimuth_min_deg: float = -90.0
    column_azimuth_max_deg: float = 90.0
    column_row_weights: tuple[float, ...] = (0.90, 1.00, 0.85)
    risk_proximity_weight: float = 0.65
    risk_ttc_weight: float = 0.35
    near_priority_distance_m: float = 0.50
    ttc_priority_horizon_s: float = 5.0
    ttc_volume_weight: float = 0.60
    distance_volume_weight: float = 0.40

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("rows and cols must be positive")
        if not 0.0 <= self.depth_percentile <= 100.0:
            raise ValueError("depth_percentile must be in range [0, 100]")
        if self.ttc_min_depth_delta_m < 0.0:
            raise ValueError("ttc_min_depth_delta_m must be non-negative")
        if self.ttc_min_dt_s <= 0.0:
            raise ValueError("ttc_min_dt_s must be > 0")
        if not 0.0 <= self.ttc_approach_ema_alpha <= 1.0:
            raise ValueError("ttc_approach_ema_alpha must be in range [0, 1]")
        if self.ttc_confirm_frames <= 0:
            raise ValueError("ttc_confirm_frames must be positive")
        if self.ttc_clear_frames <= 0:
            raise ValueError("ttc_clear_frames must be positive")
        if self.ttc_max_speed_mps <= 0.0:
            raise ValueError("ttc_max_speed_mps must be > 0")
        if len(self.column_row_weights) != self.rows:
            weights = [1.0] * self.rows
            object.__setattr__(self, "column_row_weights", tuple(weights))


@dataclass
class _CellHistory:
    depth_m: Optional[float] = None
    timestamp_s: Optional[float] = None
    approach_speed_ema_mps: float = 0.0
    approach_confirm_streak: int = 0
    non_approach_streak: int = 0


class NavigationProcessor:
    def __init__(self, config: Optional[NavigationProcessorConfig] = None) -> None:
        self.config = config or NavigationProcessorConfig()
        self._cell_history = {
            (row, col): _CellHistory()
            for row in range(self.config.rows)
            for col in range(self.config.cols)
        }
        self._frame_index = 0
        self._cached_ground_plane: Optional[GroundPlaneEstimate] = None

    def process_bundle(self, bundle: FrameBundle) -> NavigationFrameAnalysis:
        depth_m = bundle.depth.image.astype(np.float32) * float(bundle.depth.depth_scale)
        valid_mask = np.isfinite(depth_m)
        valid_mask &= depth_m >= self.config.min_depth_m
        valid_mask &= depth_m <= self.config.max_depth_m

        gravity_unit = self._estimate_gravity_unit(bundle)
        full_points, _ = self._depth_to_points(
            depth_m=depth_m,
            intrinsics=bundle.depth.intrinsics,
            downsample_step=1,
        )
        sampled_mask = np.zeros_like(valid_mask, dtype=bool)
        step = max(1, int(self.config.downsample_step))
        sampled_mask[::step, ::step] = True

        plane = self._select_ground_plane(
            points=full_points,
            sampled_mask=sampled_mask,
            valid_mask=valid_mask,
            gravity_unit=gravity_unit,
        )

        ground_mask = np.zeros_like(valid_mask, dtype=bool)
        obstacle_mask = valid_mask.copy()
        height_above_ground_m: Optional[np.ndarray] = None

        if plane is not None:
            signed_distance = self._plane_signed_distance(full_points, plane.normal, plane.offset)
            height_above_ground_m = signed_distance.astype(np.float32)
            ground_mask = valid_mask & (np.abs(height_above_ground_m) <= self.config.floor_clearance_m)
            obstacle_mask = valid_mask & (
                (height_above_ground_m > self.config.floor_clearance_m)
                | (height_above_ground_m < -self.config.dropoff_clearance_m)
            )

        percentile_depth_grid = np.full((self.config.rows, self.config.cols), np.nan, dtype=np.float32)
        ttc_grid = np.full((self.config.rows, self.config.cols), np.nan, dtype=np.float32)
        risk_grid = np.zeros((self.config.rows, self.config.cols), dtype=np.float32)
        cell_states: list[NavigationCellState] = []

        row_edges = np.linspace(0, depth_m.shape[0], self.config.rows + 1, dtype=int)
        col_edges = np.linspace(0, depth_m.shape[1], self.config.cols + 1, dtype=int)
        timestamp_s = float(bundle.depth.host_timestamp_s)

        for row in range(self.config.rows):
            for col in range(self.config.cols):
                row_slice = slice(row_edges[row], row_edges[row + 1])
                col_slice = slice(col_edges[col], col_edges[col + 1])
                cell_obstacle_mask = obstacle_mask[row_slice, col_slice]
                cell_valid_mask = valid_mask[row_slice, col_slice]
                cell_depths = depth_m[row_slice, col_slice][cell_obstacle_mask]
                sample_count = int(cell_depths.size)
                valid_count = int(np.count_nonzero(cell_valid_mask))
                obstacle_fraction = sample_count / max(1, valid_count)
                percentile_depth = self._depth_percentile(cell_depths)
                approach_speed_mps, ttc_s = self._update_ttc(
                    row=row,
                    col=col,
                    timestamp_s=timestamp_s,
                    percentile_depth_m=percentile_depth,
                )
                risk_score = self._compute_risk(
                    percentile_depth_m=percentile_depth,
                    ttc_s=ttc_s,
                    sample_count=sample_count,
                    obstacle_fraction=obstacle_fraction,
                )

                if percentile_depth is not None:
                    percentile_depth_grid[row, col] = percentile_depth
                if ttc_s is not None:
                    ttc_grid[row, col] = ttc_s
                risk_grid[row, col] = risk_score
                cell_states.append(
                    NavigationCellState(
                        row=row,
                        col=col,
                        sample_count=sample_count,
                        obstacle_fraction=obstacle_fraction,
                        percentile_depth_m=percentile_depth,
                        approach_speed_mps=approach_speed_mps,
                        ttc_s=ttc_s,
                        risk_score=risk_score,
                    )
                )

        column_states = self._build_column_states(cell_states=cell_states)
        self._frame_index += 1
        return NavigationFrameAnalysis(
            timestamp_s=timestamp_s,
            depth_m=depth_m,
            valid_mask=valid_mask,
            ground_mask=ground_mask,
            obstacle_mask=obstacle_mask,
            height_above_ground_m=height_above_ground_m,
            gravity_unit=gravity_unit,
            ground_plane=plane,
            cell_states=tuple(cell_states),
            column_states=tuple(column_states),
            depth_percentile=self.config.depth_percentile,
            percentile_depth_grid_m=percentile_depth_grid,
            ttc_grid_s=ttc_grid,
            risk_grid=risk_grid,
        )

    def _select_ground_plane(
        self,
        *,
        points: np.ndarray,
        sampled_mask: np.ndarray,
        valid_mask: np.ndarray,
        gravity_unit: Optional[np.ndarray],
    ) -> Optional[GroundPlaneEstimate]:
        interval = max(1, int(self.config.ground_plane_refit_interval_frames))
        should_refit = self._cached_ground_plane is None or (self._frame_index % interval == 0)
        if should_refit:
            self._cached_ground_plane = self._estimate_ground_plane(
                points=points,
                sampled_mask=sampled_mask,
                valid_mask=valid_mask,
                gravity_unit=gravity_unit,
            )
        return self._cached_ground_plane

    def _estimate_gravity_unit(self, bundle: FrameBundle) -> Optional[np.ndarray]:
        accel = bundle.latest_accel
        if accel is None:
            return None

        vector = accel.xyz.astype(np.float64)
        norm = float(np.linalg.norm(vector))
        if norm < 1e-6:
            return None
        return (vector / norm).astype(np.float32)

    @staticmethod
    def _depth_to_points(
        *,
        depth_m: np.ndarray,
        intrinsics: CameraIntrinsics,
        downsample_step: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        step = max(1, int(downsample_step))
        sampled_mask = np.zeros_like(depth_m, dtype=bool)
        sampled_mask[::step, ::step] = True

        ys, xs = np.indices(depth_m.shape, dtype=np.float32)
        z = depth_m
        x = (xs - float(intrinsics.ppx)) * z / float(intrinsics.fx)
        y = (ys - float(intrinsics.ppy)) * z / float(intrinsics.fy)
        return np.stack((x, y, z), axis=-1), sampled_mask

    def _estimate_ground_plane(
        self,
        *,
        points: np.ndarray,
        sampled_mask: np.ndarray,
        valid_mask: np.ndarray,
        gravity_unit: Optional[np.ndarray],
    ) -> Optional[GroundPlaneEstimate]:
        height = points.shape[0]
        start_row = int(height * (1.0 - self.config.floor_candidate_fraction))
        candidate_mask = sampled_mask & valid_mask
        candidate_mask[:start_row, :] = False
        candidate_points = points[candidate_mask]
        if candidate_points.shape[0] < max(3, self.config.min_plane_inliers):
            return None

        gravity_vec = gravity_unit.astype(np.float64) if gravity_unit is not None else None
        if gravity_vec is None:
            gravity_vec = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        up_axis = -gravity_vec / max(np.linalg.norm(gravity_vec), 1e-6)

        rng = np.random.default_rng()
        best_normal: Optional[np.ndarray] = None
        best_offset: Optional[float] = None
        best_inliers: Optional[np.ndarray] = None
        best_score = -1.0

        cos_tilt = math.cos(math.radians(self.config.max_plane_tilt_deg))

        for _ in range(self.config.ransac_iterations):
            sample_indices = rng.choice(candidate_points.shape[0], size=3, replace=False)
            p0, p1, p2 = candidate_points[sample_indices]
            normal = np.cross(p1 - p0, p2 - p0)
            norm = float(np.linalg.norm(normal))
            if norm < 1e-6:
                continue

            normal = normal / norm
            alignment = float(abs(np.dot(normal, up_axis)))
            if alignment < self.config.gravity_alignment_cos_min or alignment < cos_tilt:
                continue

            if float(np.dot(normal, up_axis)) < 0.0:
                normal = -normal

            offset = -float(np.dot(normal, p0))
            distances = np.abs(self._plane_signed_distance(candidate_points, normal, offset))
            inliers = distances <= self.config.ground_inlier_threshold_m
            inlier_count = int(np.count_nonzero(inliers))
            if inlier_count < self.config.min_plane_inliers:
                continue

            score = float(inlier_count) - float(np.mean(distances[inliers])) * 100.0
            if score > best_score:
                best_score = score
                best_normal = normal.astype(np.float32)
                best_offset = offset
                best_inliers = inliers

        if best_normal is None or best_offset is None or best_inliers is None:
            return None

        fitted_points = candidate_points[best_inliers]
        refined_normal, refined_offset = self._refine_plane(fitted_points, up_axis=up_axis)
        residuals = np.abs(self._plane_signed_distance(fitted_points, refined_normal, refined_offset))
        return GroundPlaneEstimate(
            normal=refined_normal.astype(np.float32),
            offset=float(refined_offset),
            inlier_count=int(fitted_points.shape[0]),
            mean_residual_m=float(np.mean(residuals)),
        )

    @staticmethod
    def _refine_plane(points: np.ndarray, *, up_axis: np.ndarray) -> tuple[np.ndarray, float]:
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1]
        if float(np.dot(normal, up_axis)) < 0.0:
            normal = -normal
        normal = normal / max(np.linalg.norm(normal), 1e-6)
        offset = -float(np.dot(normal, centroid))
        return normal, offset

    @staticmethod
    def _plane_signed_distance(points: np.ndarray, normal: np.ndarray, offset: float) -> np.ndarray:
        return np.tensordot(points, normal, axes=([-1], [0])) + float(offset)

    def _depth_percentile(self, depth_values_m: np.ndarray) -> Optional[float]:
        if depth_values_m.size == 0:
            return None
        percentile = float(np.clip(self.config.depth_percentile, 0.0, 100.0))
        if percentile <= 0.0:
            return float(np.min(depth_values_m))
        if percentile >= 100.0:
            return float(np.max(depth_values_m))

        index = int((percentile / 100.0) * (depth_values_m.size - 1))
        partitioned = np.partition(depth_values_m, index)
        return float(partitioned[index])

    def _update_ttc(
        self,
        *,
        row: int,
        col: int,
        timestamp_s: float,
        percentile_depth_m: Optional[float],
    ) -> tuple[float, Optional[float]]:
        history = self._cell_history[(row, col)]
        approach_speed_mps = 0.0
        ttc_s: Optional[float] = None

        if percentile_depth_m is None:
            history.depth_m = None
            history.timestamp_s = timestamp_s
            history.approach_speed_ema_mps = 0.0
            history.approach_confirm_streak = 0
            history.non_approach_streak = 0
            return approach_speed_mps, ttc_s

        if (
            history.depth_m is not None
            and history.timestamp_s is not None
            and timestamp_s > history.timestamp_s
        ):
            dt = timestamp_s - history.timestamp_s
            if dt >= self.config.ttc_min_dt_s:
                depth_delta_m = history.depth_m - percentile_depth_m
                if depth_delta_m >= self.config.ttc_min_depth_delta_m:
                    instant_speed_mps = depth_delta_m / dt
                    instant_speed_mps = float(
                        np.clip(instant_speed_mps, 0.0, self.config.ttc_max_speed_mps)
                    )
                    alpha = float(np.clip(self.config.ttc_approach_ema_alpha, 0.0, 1.0))
                    history.approach_speed_ema_mps = (
                        (1.0 - alpha) * history.approach_speed_ema_mps
                        + alpha * instant_speed_mps
                    )
                    history.approach_confirm_streak += 1
                    history.non_approach_streak = 0
                else:
                    history.approach_confirm_streak = 0
                    history.non_approach_streak += 1
                    if history.non_approach_streak >= self.config.ttc_clear_frames:
                        history.approach_speed_ema_mps = 0.0

                approach_speed_mps = history.approach_speed_ema_mps
                if (
                    history.approach_confirm_streak >= self.config.ttc_confirm_frames
                    and approach_speed_mps >= self.config.ttc_min_speed_mps
                ):
                    ttc_s = percentile_depth_m / approach_speed_mps

        history.depth_m = percentile_depth_m
        history.timestamp_s = timestamp_s
        return approach_speed_mps, ttc_s

    def _compute_risk(
        self,
        *,
        percentile_depth_m: Optional[float],
        ttc_s: Optional[float],
        sample_count: int,
        obstacle_fraction: float,
    ) -> float:
        if percentile_depth_m is None:
            return 0.0
        if sample_count < self.config.min_obstacle_points_per_cell:
            return 0.0

        proximity_span = max(1e-6, self.config.proximity_far_m - self.config.proximity_near_m)
        proximity_score = 1.0 - ((percentile_depth_m - self.config.proximity_near_m) / proximity_span)
        proximity_score = float(np.clip(proximity_score, 0.0, 1.0))

        if ttc_s is None or not np.isfinite(ttc_s):
            ttc_score = 0.0
        else:
            ttc_score = 1.0 - (ttc_s / self.config.ttc_horizon_s)
            ttc_score = float(np.clip(ttc_score, 0.0, 1.0))

        occupancy_score = min(1.0, sample_count / float(max(1, self.config.occupancy_denominator)))
        occupancy_score = max(occupancy_score, float(np.clip(obstacle_fraction, 0.0, 1.0)))

        blended = (
            self.config.risk_proximity_weight * proximity_score
            + self.config.risk_ttc_weight * ttc_score
        )
        return float(np.clip(blended * occupancy_score, 0.0, 1.0))

    def _build_column_states(
        self,
        *,
        cell_states: list[NavigationCellState],
    ) -> list[NavigationColumnState]:
        grouped: dict[int, list[NavigationCellState]] = {col: [] for col in range(self.config.cols)}
        for cell_state in cell_states:
            grouped[cell_state.col].append(cell_state)

        column_states: list[NavigationColumnState] = []
        azimuths = np.linspace(
            self.config.column_azimuth_min_deg,
            self.config.column_azimuth_max_deg,
            self.config.cols,
            dtype=np.float32,
        )

        for col in range(self.config.cols):
            column_cells = grouped[col]
            weighted_risks = []
            for cell in column_cells:
                weight = self.config.column_row_weights[cell.row]
                weighted_risks.append(cell.risk_score * weight)

            column_risk = float(max(weighted_risks, default=0.0))
            best_cell = max(
                column_cells,
                key=lambda cell: cell.risk_score * self.config.column_row_weights[cell.row],
                default=None,
            )

            depth_m = best_cell.percentile_depth_m if best_cell is not None else None
            ttc_s = best_cell.ttc_s if best_cell is not None else None
            sample_count = sum(cell.sample_count for cell in column_cells)
            pitch_hz = self._risk_to_pitch(depth_m, column_risk)
            pulse_hz = self._risk_to_pulse_rate(ttc_s, column_risk)
            volume = self._risk_to_volume(depth_m=depth_m, ttc_s=ttc_s, risk_score=column_risk)
            column_states.append(
                NavigationColumnState(
                    col=col,
                    azimuth_deg=float(azimuths[col]),
                    sample_count=sample_count,
                    risk_score=column_risk,
                    percentile_depth_m=depth_m,
                    ttc_s=ttc_s,
                    pitch_hz=pitch_hz,
                    pulse_hz=pulse_hz,
                    volume=volume,
                )
            )

        return column_states

    def _risk_to_pitch(self, depth_m: Optional[float], risk_score: float) -> float:
        if depth_m is None or risk_score <= 0.0:
            return self.config.pitch_hz_far

        proximity_span = max(1e-6, self.config.proximity_far_m - self.config.proximity_near_m)
        proximity_score = 1.0 - ((depth_m - self.config.proximity_near_m) / proximity_span)
        proximity_score = float(np.clip(proximity_score, 0.0, 1.0))
        return float(
            self.config.pitch_hz_far
            + proximity_score * (self.config.pitch_hz_near - self.config.pitch_hz_far)
        )

    def _risk_to_pulse_rate(self, ttc_s: Optional[float], risk_score: float) -> float:
        if risk_score <= 0.0:
            return 0.0
        if ttc_s is None or not np.isfinite(ttc_s):
            return self.config.pulse_rate_hz_min

        urgency = 1.0 - float(np.clip(ttc_s / self.config.ttc_horizon_s, 0.0, 1.0))
        return float(
            self.config.pulse_rate_hz_min
            + urgency * (self.config.pulse_rate_hz_max - self.config.pulse_rate_hz_min)
        )

    def _risk_to_volume(self, *, depth_m: Optional[float], ttc_s: Optional[float], risk_score: float) -> float:
        if risk_score <= 0.0:
            return 0.0

        if depth_m is None:
            distance_factor = 0.0
        else:
            distance_factor = 1.0 - float(np.clip(depth_m / self.config.near_priority_distance_m, 0.0, 1.0))

        if ttc_s is None or not np.isfinite(ttc_s):
            ttc_factor = 0.0
        else:
            ttc_factor = 1.0 - float(np.clip(ttc_s / self.config.ttc_priority_horizon_s, 0.0, 1.0))

        urgency = (
            self.config.ttc_volume_weight * ttc_factor
            + self.config.distance_volume_weight * distance_factor
        )
        urgency = float(np.clip(urgency, 0.0, 1.0))

        # Keep stationary hazards audible at close range while still letting risk gate noise.
        near_floor = distance_factor if depth_m is not None and depth_m <= self.config.near_priority_distance_m else 0.0
        volume = max(near_floor, urgency) * float(np.clip(risk_score, 0.0, 1.0))
        return float(np.clip(volume, 0.0, 1.0))