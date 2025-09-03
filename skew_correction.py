#!/usr/bin/env python3
# Camera based skew calibration for Klipper
#
# Copyright (C) 2025 Marius Wachtler <undingen@gmail.com>
#
# This file may be distributed under the terms of the GNU GPLv3 license.

"""
This script provides a comprehensive solution for calibrating and correcting XY
skew on Klipper-based 3D printers using a camera mounted on the toolhead.

It automates the entire workflow:
  1. Board Generation: Creates a printable ChArUco calibration board, centered
     on both A4 and Letter paper sizes. A ChArUco board combines a chessboard
     pattern with ArUco markers, allowing for highly accurate and robust corner
     detection.

  2. Guided Image Capture for Camera Calibration: Provides a structured routine
     for capturing images needed for camera calibration. It moves the toolhead
     to various XY positions and Z heights to ensure a diverse set of perspectives,
     which is critical for an accurate calibration.

  3. Camera Intrinsics Calibration: Processes the captured images to determine
     the camera's intrinsic properties, such as focal length, principal point,
     and lens distortion. These are saved to `K.npy` and `dist.npy`.

  4. Automated Skew Calculation: Uses the calibrated camera to precisely measure
     the printer's movement. It automatically determines the camera-to-nozzle
     offset and then calculates the affine transformation that maps commanded
     printer coordinates to the actual positions observed on the bed. From this,
     it derives the `SET_SKEW` parameters required by Klipper.

This toolkit simplifies a complex process, making accurate skew correction
accessible without manual measurements or tedious calculations.

--------------------------------------------------------------------------------
Usage Workflow:
--------------------------------------------------------------------------------

Step 1: Generate and Print the Calibration Board
  - Run the script to create the PDF files.
  - Print `charuco_A4_5mm.pdf` or `charuco_Letter_5mm.pdf` at **100% scale**
    (Actual Size). Do not use "Fit to Page".
  - Securely fasten the printed board to the printer's bed.

  $ python3 skew_correction.py generate-board

Step 2: Capture Images for Camera Calibration
  - This guided process will move the toolhead and capture images from various
    viewpoints. For best results, it will ask you to slightly tilt the bed
    between phases to introduce more perspective diversity.
  - Images are saved directly into the specified directory (default: `camera_calibration_imgs`).

  $ python3 skew_correction.py capture-calibration-images --camera-id http://klipper.local/webcam/?action=snapshot

Step 3: Calibrate the Camera
  - This step processes the images from Step 2 to calculate and save the camera's
    intrinsic parameters (`K.npy` and `dist.npy`).
  - Use the path to the images saved in the previous step.

  $ python3 skew_correction.py calibrate-camera "camera_calibration_imgs/*.jpg"

Step 4: Calculate Printer Skew
  - The final step. The script will move the toolhead over the board, capture
    images, and perform calculations to find the camera-nozzle offset and
    the printer's skew.
  - It will output the `SET_SKEW` command ready to be used in Klipper.

  $ python3 skew_correction.py skew --z 40

--------------------------------------------------------------------------------
Dependencies:
  - opencv-contrib-python (for ArUco/ChArUco support)
  - numpy
  - requests
  - Optional for PDF generation: Pillow, img2pdf, pikepdf

Install with:
  pip install opencv-contrib-python numpy requests 
  Optional: pip install Pillow img2pdf pikepdf


Test skews (for tests):
   0.5°: SET_SKEW XY=142.037,140.803,100.000
   1.0°: SET_SKEW XY=142.650,140.182,100.000
   1.5°: SET_SKEW XY=143.260,139.558,100.000
  -2.0°: SET_SKEW XY=138.932,143.868,100.000
 -10.0°: SET_SKEW XY=128.558,153.209,100.000
"""

import argparse
import glob
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2 as cv
import numpy as np
import requests

# --- Configuration Constants ---

# Board geometry (identical for A4 & Letter)
SQUARE_MM = 5.0
SQUARES_X = 38  # columns
SQUARES_Y = 52  # rows
MARKER_RATIO = 0.72  # marker_length = ratio * square_length
DICT_ID = cv.aruco.DICT_5X5_1000

# Skew calibration tuning parameters
FRAMES_PER_POSE = 4  # Min accepted frames per pose
REQUIRED_CORNERS = 16  # Min ChArUco corners per frame for a valid detection
MAX_REPROJECTION_ERROR_PX = 2.0  # Reject frames with RMS reprojection error > this
DWELL_AFTER_MOVE_SEC = 1.0  # Wait time after motion before capture
MAX_MEDIAN_XY_RESIDUAL_MM = 1.0  # Abort if fit quality is poor
WARN_RMS_ABOVE_PX = 1.0  # Warn if intrinsics calibration RMS is high


class MoonrakerController:
    """Handles all communication with the printer via the Moonraker API."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._bounds: Optional[Dict[str, float]] = None
        print(f"[info] Moonraker configured for URL: {self.base_url}")

    @property
    def bounds(self) -> Dict[str, float]:
        """Fetch and cache toolhead (bed) bounds. Raises on failure."""
        if self._bounds is None:
            try:
                url = f"{self.base_url}/printer/objects/query?toolhead"
                resp = requests.get(url, timeout=5)
                resp.raise_for_status()
                data = resp.json()["result"]["status"]["toolhead"]
                self._bounds = {
                    'x_min': data["axis_minimum"][0],
                    'y_min': data["axis_minimum"][1],
                    'z_min': data["axis_minimum"][2],
                    'x_max': data["axis_maximum"][0],
                    'y_max': data["axis_maximum"][1],
                    'z_max': data["axis_maximum"][2],
                }
                print(
                    f"[info] Bed bounds: "
                    f"X({self._bounds['x_min']:.1f}, {self._bounds['x_max']:.1f}), "
                    f"Y({self._bounds['y_min']:.1f}, {self._bounds['y_max']:.1f}), "
                    f"Z({self._bounds['z_min']:.1f}, {self._bounds['z_max']:.1f})"
                )
            except Exception as e:
                print(f"[error] Failed to get printer bounds: {e}", file=sys.stderr)
                raise
        return self._bounds

    def get_bed_size(self, margin: float = 0.0) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Return ((x_min, y_min), (x_max, y_max)) with an applied margin."""
        b = self.bounds
        return (b['x_min'] + margin, b['y_min'] + margin), (b['x_max'] - margin, b['y_max'] - margin)

    def move_head(self, x: float, y: float, z: float, feedrate_mm_per_min: int = 3000) -> None:
        """Move toolhead to (x, y, z) within clamped bounds."""
        b = self.bounds
        x_clamped = max(b['x_min'], min(x, b['x_max']))
        y_clamped = max(b['y_min'], min(y, b['y_max']))
        z_clamped = max(b['z_min'], min(z, b['z_max']))

        gcode = f"G1 X{x_clamped:.3f} Y{y_clamped:.3f} Z{z_clamped:.3f} F{feedrate_mm_per_min}\nM400"
        try:
            url = f"{self.base_url}/printer/gcode/script"
            r = requests.post(url, params={"script": gcode}, timeout=15)
            r.raise_for_status()
        except Exception as e:
            print(f"[error] move_head failed: {e}", file=sys.stderr)
            raise


class Camera:
    """Manages camera interactions."""

    def __init__(self, camera_id: str):
        self.camera_id = camera_id

    def _capture_frame(self, rotate=False):
        frame = None

        try:
            response = requests.get(self.camera_id, timeout=1)
            response.raise_for_status()
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            frame = cv.imdecode(img_array, cv.IMREAD_COLOR)
        except requests.RequestException as e:
            pass

        if frame is None:
            return None

        # Flip the image horizontally and vertically
        frame = cv.flip(frame, -1)
        if rotate:
            # Rotate the image 90 degrees counterclockwise
            frame = cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)

        return frame


    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from the camera source."""
        retry=5
        for _ in range(retry):
            frame = self._capture_frame(rotate=True)
            if frame is not None:
                return frame
            
        print(f"  [error] Failed to capture frame from camera '{self.camera_id}'")
        return None


class CharucoBoard:
    """Defines and builds the ChArUco board."""

    def __init__(self):
        self.square_mm = SQUARE_MM
        self.squares_x = SQUARES_X
        self.squares_y = SQUARES_Y
        self.marker_ratio = MARKER_RATIO
        self.dict_id = DICT_ID
        print(f"[info] Using {self.square_mm} mm squares for ChArUco board.")
        self.board = self._build()

    def _build(self) -> cv.aruco.CharucoBoard:
        """Create the CharucoBoard object."""
        dictionary = cv.aruco.getPredefinedDictionary(self.dict_id)
        return cv.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            float(self.square_mm),
            float(self.square_mm * self.marker_ratio),
            dictionary,
        )

    def generate_image(self, dpi: int = 300) -> np.ndarray:
        """Generate the board image at a specified DPI."""
        def mm_to_px(mm: float) -> int:
            return int(round(mm / 25.4 * dpi))

        width_px = mm_to_px(self.squares_x * self.square_mm)
        height_px = mm_to_px(self.squares_y * self.square_mm)
        return self.board.generateImage((width_px, height_px), 0, 1)

    @staticmethod
    def create_detector(
        board: cv.aruco.CharucoBoard,
        K: Optional[np.ndarray] = None,
        dist: Optional[np.ndarray] = None
    ) -> cv.aruco.CharucoDetector:
        """Create a CharucoDetector with optional intrinsics."""
        ch_params = cv.aruco.CharucoParameters()
        if K is not None and dist is not None:
            ch_params.cameraMatrix = K
            ch_params.distCoeffs = dist

        det_params = cv.aruco.DetectorParameters()
        det_params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
        det_params.cornerRefinementWinSize = 5
        det_params.cornerRefinementMaxIterations = 30
        det_params.cornerRefinementMinAccuracy = 0.01

        return cv.aruco.CharucoDetector(board, ch_params, det_params)


class PDFGenerator:
    """Generates centered, unscaled PDFs of the ChArUco board."""

    PAPER_SIZES_MM = {
        "A4": (210.0, 297.0),
        "LETTER": (215.9, 279.4),
    }

    def __init__(self, board: CharucoBoard):
        self.board = board

    def generate_pdfs(self) -> None:
        """Generate and save A4 and Letter PDFs."""
        img = self.board.generate_image()
        board_dims_mm = (
            self.board.squares_x * self.board.square_mm,
            self.board.squares_y * self.board.square_mm,
        )
        print(f"Board is {board_dims_mm[0]:.1f} mm x {board_dims_mm[1]:.1f} mm")

        for paper_name in self.PAPER_SIZES_MM:
            filename = f"charuco_{paper_name}_{int(self.board.square_mm)}mm.pdf"
            self._img_to_pdf(img, paper_name, filename, board_dims_mm)

    def _img_to_pdf(self, img: np.ndarray, paper_name: str, out_pdf: str, board_dims_mm: Tuple[float, float]) -> None:
        """Writes a single-page PDF with the board image centered."""
        try:
            import img2pdf
            import pikepdf
        except ImportError as e:
            raise RuntimeError("PDF export requires: pip install img2pdf pikepdf") from e

        page_w_mm, page_h_mm = self.PAPER_SIZES_MM[paper_name.upper()]

        def mm_to_pt(mm: float) -> float:
            return mm * 72.0 / 25.4

        page_w_pt, page_h_pt = mm_to_pt(page_w_mm), mm_to_pt(page_h_mm)
        img_w_pt, img_h_pt = mm_to_pt(board_dims_mm[0]), mm_to_pt(board_dims_mm[1])

        if img_w_pt > page_w_pt or img_h_pt > page_h_pt:
            raise ValueError("Board is larger than the paper. Reduce squares or square size.")

        border_x = (page_w_pt - img_w_pt) / 2.0
        border_y = (page_h_pt - img_h_pt) / 2.0

        layout = img2pdf.get_layout_fun(
            pagesize=(page_w_pt, page_h_pt),
            imgsize=((img2pdf.ImgSize.abs, img_w_pt), (img2pdf.ImgSize.abs, img_h_pt)),
            border=(border_x, border_y),
            auto_orient=False,
        )

        _, buf = cv.imencode(".png", img)
        with open(out_pdf, "wb") as f:
            f.write(img2pdf.convert(bytes(buf), layout_fun=layout))

        # Set viewer preferences for "Actual Size" printing
        with pikepdf.open(out_pdf, allow_overwriting_input=True) as pdf:
            vp = pdf.Root.get("/ViewerPreferences", pikepdf.Dictionary())
            vp["/PrintScaling"] = pikepdf.Name("/None")
            pdf.Root["/ViewerPreferences"] = vp
            pdf.save(out_pdf)

        print(f"[ok] Wrote {out_pdf}")


@dataclass
class PoseObservation:
    """Stores a single, high-quality observation of the board's pose."""
    p_cmd: Tuple[float, float]  # Commanded printer XY (mm)
    r_mat: np.ndarray  # 3x3 rotation matrix (board -> camera)
    t_vec: np.ndarray  # 3x1 translation vector (board -> camera)
    reprojection_rms: float


class IntrinsicsCalibrator:
    """Calibrates camera intrinsics from a set of images."""

    def __init__(self, board: CharucoBoard):
        self.board = board.board
        self.detector = CharucoBoard.create_detector(self.board)

    def calibrate_from_images(self, images_glob: str) -> None:
        """
        Processes images, calibrates the camera, and saves the results.
        Raises RuntimeError on failure.
        """
        paths = sorted(glob.glob(images_glob))
        if not paths:
            raise FileNotFoundError(f"No images match '{images_glob}'")

        all_corners, all_ids, imsize = self._process_images(paths)

        if len(all_ids) < 12:
            raise RuntimeError(
                f"Only {len(all_ids)} good views found. At least 12 are recommended. "
                "Capture more images with varied angles and ensure good focus."
            )

        print("\n[info] Calibrating camera intrinsics from collected corners...")
        rms, K, dist, _, _ = cv.aruco.calibrateCameraCharuco(
            all_corners, all_ids, self.board, imsize, None, None, flags=0
        )

        np.save("K.npy", K)
        np.save("dist.npy", dist)

        self._print_calibration_summary(rms, K, imsize)
        print("\n[ok] Saved camera intrinsics to K.npy and dist.npy")

    def _process_images(self, paths: List[str]) -> Tuple[List, List, Tuple[int, int]]:
        """Detects ChArUco corners in a list of image files."""
        all_corners, all_ids = [], []
        imsize = None

        for idx, p in enumerate(paths, 1):
            short_name = os.path.basename(p)
            print(f"[{idx}/{len(paths)}] Processing {short_name}...")
            img = cv.imread(p, cv.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  [warn] Skipping unreadable image.")
                continue

            if imsize is None:
                imsize = (img.shape[1], img.shape[0])

            ch_corners, ch_ids, _, _ = self.detector.detectBoard(img)
            n_corners = 0 if ch_ids is None else len(ch_ids)

            if n_corners < REQUIRED_CORNERS * 2:
                print(f"  [skip] Found {n_corners} corners (less than required {REQUIRED_CORNERS*2}).")
                continue

            all_corners.append(ch_corners)
            all_ids.append(ch_ids)
            print(f"  [keep] Found {n_corners} corners.")

        return all_corners, all_ids, imsize

    def _print_calibration_summary(self, rms: float, K: np.ndarray, imsize: Tuple[int, int]) -> None:
        """Displays a human-friendly summary of the calibration results."""
        w, h = imsize
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        fovx = 2 * np.degrees(np.arctan(w / (2 * fx)))
        fovy = 2 * np.degrees(np.arctan(h / (2 * fy)))

        print(f"\n[ok] RMS reprojection error: {rms:.4f} px")
        if rms > WARN_RMS_ABOVE_PX:
            print(
                f"  [warn] RMS error is high (>{WARN_RMS_ABOVE_PX:.1f} px). "
                "For better results, ensure the board fills more of the frame, "
                "use good lighting, and provide 30-60 sharp, diverse views."
            )

        print("[ok] Intrinsics (pixels):")
        print(f"  fx={fx:.2f}, fy={fy:.2f}")
        print(f"  cx={cx:.2f}, cy={cy:.2f}")
        print("[ok] Field of view (degrees):")
        print(f"  Horizontal={fovx:.2f}, Vertical={fovy:.2f}")


class SkewCalibrator:
    """
    Orchestrates the automated skew calibration process.

    This class solves for the physical relationship between the printer's
    commanded coordinate system and the camera's view of the calibration board.
    It models this relationship as:

        R^T * (d - t) = [ A*P + b ]
                        [    h    ]

    Where:
    - P: Commanded 2D printer position (X, Y).
    - R, t: Rotation and translation from the board's coordinate system to the
            camera's, as measured by OpenCV.
    - d: The unknown 3D vector from the camera's optical center to the nozzle tip.
    - A, b: An unknown 2D affine transform (a 2x2 matrix A and 2x1 vector b) that
            maps commanded printer coordinates (P) to the board's coordinate system.
            This transform accounts for skew, scaling, and origin offsets.
    - h: The unknown height of the camera above the board in the board's frame.

    The script captures multiple (P, R, t) observations from different toolhead
    positions and solves for the unknown parameters (d, A, b, h) using a
    weighted linear least-squares fit.
    """

    def __init__(self, camera: Camera, moonraker: MoonrakerController, board: CharucoBoard):
        self.camera = camera
        self.moonraker = moonraker
        self.charuco_board = board

    def run_skew_calibration(self, z_height: float, margin: float) -> None:
        """Main entry point for the skew calibration process."""
        try:
            K = np.load("K.npy")
            dist = np.load("dist.npy")
        except FileNotFoundError:
            raise RuntimeError(
                "Intrinsics files K.npy and/or dist.npy not found. "
                "Please run the 'calibrate-camera' command first."
            )

        detector = CharucoBoard.create_detector(self.charuco_board.board, K, dist)

        print("\n[info] Starting skew calibration process...")
        print("[info] This will automatically find the camera-nozzle offset.")

        # Step 1: Bootstrap phase to get an initial estimate
        print("\n--- Phase 1: Bootstrap Sampling (near bed center) ---")
        bootstrap_samples = self._collect_samples_at_points(
            self._get_bootstrap_positions(margin), detector, K, dist, z_height
        )
        if len(bootstrap_samples) < 6:
            raise RuntimeError(f"Bootstrap failed: only got {len(bootstrap_samples)} samples. Need >= 6.")

        # Step 2: Robust initial fit
        params, kept_samples = self._robust_fit(bootstrap_samples)

        # Step 3: Grid phase for wider coverage and refinement
        print("\n--- Phase 2: Grid Sampling (for refinement) ---")
        grid_positions = self._plan_grid_positions(params, margin)
        grid_samples = self._collect_samples_at_points(
            grid_positions, detector, K, dist, z_height
        )

        # Step 4: Final fit and result calculation
        all_samples = kept_samples + grid_samples
        if len(all_samples) < 12:
            print(f"[warn] Low sample count ({len(all_samples)}). Results may be less accurate.")

        final_params, final_samples = self._robust_fit(all_samples, drop_ratio=0.1) # Less aggressive drop
        self._analyze_and_print_results(final_params, final_samples, margin)

    def _get_bootstrap_positions(self, margin: float) -> List[Tuple[float, float]]:
        """Defines a 5-point pattern near the bed center for initial estimation."""
        (xmin, ymin), (xmax, ymax) = self.moonraker.get_bed_size(margin=margin)
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        offset = max(15.0, min(30.0, 0.08 * min(xmax - xmin, ymax - ymin)))
        return [
            (cx, cy),
            (cx + offset, cy), (cx - offset, cy),
            (cx, cy + offset), (cx, cy - offset),
        ]

    def _plan_grid_positions(self, params: Dict[str, np.ndarray], margin: float) -> List[Tuple[float, float]]:
        """Uses the initial fit to plan sample points in a grid on the board."""
        A, b = params['A'], params['b']
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            raise RuntimeError("Initial transformation matrix 'A' is singular. Cannot proceed.")

        board_w = (self.charuco_board.squares_x - 1) * self.charuco_board.square_mm
        board_h = (self.charuco_board.squares_y - 1) * self.charuco_board.square_mm
        margin_mm = 10.0 # Margin within the board itself

        qxs = np.linspace(margin_mm, board_w - margin_mm, 4)
        qys = np.linspace(margin_mm, board_h - margin_mm, 4)

        (xmin, ymin), (xmax, ymax) = self.moonraker.get_bed_size(margin=margin)
        positions = []
        for qy in qys:
            for qx in qxs:
                q = np.array([[qx], [qy]])
                P_printer = A_inv @ (q - b)
                tx = np.clip(P_printer[0, 0], xmin, xmax)
                ty = np.clip(P_printer[1, 0], ymin, ymax)
                positions.append((tx, ty))
        return positions

    def _collect_samples_at_points(
        self, points: List[Tuple[float, float]], detector: cv.aruco.CharucoDetector,
        K: np.ndarray, dist: np.ndarray, z_height: float
    ) -> List[PoseObservation]:
        """Iterates through points, moves the head, and captures quality pose data."""
        all_samples = []
        for i, (x, y) in enumerate(points):
            self.moonraker.move_head(x, y, z_height)
            time.sleep(DWELL_AFTER_MOVE_SEC)
            print(f"[{i+1}/{len(points)}] Capturing at ({x:.1f}, {y:.1f})... ", end="", flush=True)

            accepted_local = []
            max_attempts = max(FRAMES_PER_POSE * 4, 20)
            for _ in range(max_attempts):
                img = self.camera.capture_frame()
                if img is None:
                    continue
                pose = self._estimate_pose(img, detector, K, dist)
                if pose and pose.reprojection_rms <= MAX_REPROJECTION_ERROR_PX:
                    accepted_local.append(pose)
                if len(accepted_local) >= FRAMES_PER_POSE:
                    break

            if accepted_local:
                # Add all accepted observations from this position
                for pose in accepted_local:
                    pose.p_cmd = (x, y)  # Ensure commanded position is stored for each
                all_samples.extend(accepted_local)
                best_rms = min(p.reprojection_rms for p in accepted_local)
                print(f"OK ({len(accepted_local)} frames, best RMS={best_rms:.3f}px)")
            else:
                print("Failed (no valid poses found)")
        return all_samples

    def _estimate_pose(
        self, img: np.ndarray, detector: cv.aruco.CharucoDetector, K: np.ndarray, dist: np.ndarray
    ) -> Optional[PoseObservation]:
        """Estimates board pose from a single image."""
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ch_corners, ch_ids, _, _ = detector.detectBoard(gray)

        if ch_ids is None or len(ch_ids) < REQUIRED_CORNERS:
            return None

        ok, rvec, tvec = cv.aruco.estimatePoseCharucoBoard(
            ch_corners, ch_ids, self.charuco_board.board, K, dist, None, None
        )
        if not ok:
            return None

        # Calculate reprojection error
        obj_pts = self.charuco_board.board.getChessboardCorners()[ch_ids.flatten()]
        img_proj, _ = cv.projectPoints(obj_pts, rvec, tvec, K, dist)
        err = np.linalg.norm(ch_corners - img_proj, axis=2)
        rms = float(np.sqrt(np.mean(err**2)))
        R, _ = cv.Rodrigues(rvec)

        # Initialize p_cmd to a placeholder; it will be set correctly in the calling function.
        return PoseObservation(p_cmd=(0.0, 0.0), r_mat=R, t_vec=tvec, reprojection_rms=rms)

    def _robust_fit(self, samples: List[PoseObservation], drop_ratio: float = 0.2) -> Tuple[Dict[str, np.ndarray], List[PoseObservation]]:
        """
        Performs a least-squares fit, removes outliers, and refits for robustness.
        Returns the solved parameters and the list of inlier samples.
        """
        params = self._solve_least_squares(samples)

        # Iterative refinement: remove worst offenders and refit
        residuals = self._calculate_xy_residuals(samples, params)
        if not residuals.size: return params, samples

        cutoff = np.quantile(residuals, 1.0 - drop_ratio)
        inliers = [s for s, r in zip(samples, residuals) if r <= cutoff]

        if len(inliers) < 6:
            print("[warn] Too few inliers after outlier rejection. Using initial fit.")
            return params, samples

        print(f"[fit] Refining fit with {len(inliers)}/{len(samples)} inlier samples.")
        refined_params = self._solve_least_squares(inliers)
        return refined_params, inliers

    def _solve_least_squares(self, samples: List[PoseObservation]) -> Dict[str, np.ndarray]:
        """
        Constructs and solves the linear system L * x = r for the 10 parameters.
        x = [d_x, d_y, d_z, a11, a12, a21, a22, b1, b2, h]^T
        """
        L_rows, r_vals = [], []
        for s in samples:
            M = s.r_mat.T
            mt = (M @ s.t_vec).flatten()
            Px, Py = s.p_cmd
            w = 1.0 / (1.0 + s.reprojection_rms**2)  # Weight by quality

            L_rows.append(w * np.array([M[0,0], M[0,1], M[0,2], -Px, -Py, 0, 0, -1, 0, 0]))
            L_rows.append(w * np.array([M[1,0], M[1,1], M[1,2], 0, 0, -Px, -Py, 0, -1, 0]))
            L_rows.append(w * np.array([M[2,0], M[2,1], M[2,2], 0, 0, 0, 0, 0, 0, -1]))
            r_vals.extend(w * mt)

        L = np.array(L_rows)
        r = np.array(r_vals).reshape(-1, 1)

        x, _, _, _ = np.linalg.lstsq(L, r, rcond=None)

        return {
            "d": x[0:3].reshape(3, 1),
            "A": np.array([[x[3,0], x[4,0]], [x[5,0], x[6,0]]]),
            "b": x[7:9].reshape(2, 1),
            "h": x[9,0],
        }

    def _calculate_xy_residuals(self, samples: List[PoseObservation], params: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculates the XY geometric error for each sample given a set of parameters."""
        d, A, b = params['d'], params['A'], params['b']
        residuals = []
        for s in samples:
            p_board_coords = (s.r_mat.T @ (d - s.t_vec)).flatten()
            p_printer_mapped = (A @ np.array(s.p_cmd).reshape(2, 1) + b).flatten()
            residuals.append(np.linalg.norm(p_board_coords[:2] - p_printer_mapped))
        return np.array(residuals)

    def _analyze_and_print_results(self, params: Dict[str, np.ndarray], samples: List[PoseObservation], margin: float) -> None:
        """Computes final skew values and prints the summary report."""
        # Final residual check
        residuals = self._calculate_xy_residuals(samples, params)
        median_xy = np.median(residuals)
        print(f"[fit] Final median XY residual = {median_xy:.3f} mm")
        if median_xy > MAX_MEDIAN_XY_RESIDUAL_MM:
             print(f"[warn] Fit quality may be suboptimal: median XY residual is {median_xy:.3f} mm.")

        # Analyze the affine transform 'A' for scale
        A_fit = params['A']
        alpha_x = np.linalg.norm(A_fit[:, 0])
        alpha_y = np.linalg.norm(A_fit[:, 1])
        alpha_mean = (alpha_x + alpha_y) / 2.0
        est_square_mm = self.charuco_board.square_mm / alpha_mean
        dev_pct = 100 * (est_square_mm / self.charuco_board.square_mm - 1.0)

        print("\n--- Calibration Results ---")
        print(f"[scale] Estimated printed square size: {est_square_mm:.3f} mm "
              f"(declared {self.charuco_board.square_mm:.1f} mm, deviation {dev_pct:+.2f}%)")
        if abs(dev_pct) > 2.0:
            print("[warn] Printed board scale seems off by >2%. Reprint at 100% scale.")

        # Calculate Klipper skew parameters
        (xmin, ymin), (xmax, ymax) = self.moonraker.get_bed_size(margin=margin)
        pA = np.array([[xmin], [ymin]])
        pB = np.array([[xmax], [ymin]])
        pC = np.array([[xmax], [ymax]])
        pD = np.array([[xmin], [ymax]])

        qA = (A_fit @ pA + params['b']).flatten()
        qB = (A_fit @ pB + params['b']).flatten()
        qC = (A_fit @ pC + params['b']).flatten()
        qD = (A_fit @ pD + params['b']).flatten()

        AD = np.linalg.norm(qA - qD)
        AC = np.linalg.norm(qA - qC)
        BD = np.linalg.norm(qB - qD)

        # Compute skew angle from diagonal lengths
        # Law of cosines: cos(theta) = (a^2 + b^2 - c^2) / (2ab)
        ab_len = np.linalg.norm(qA - qB) # Bed width in board units
        cos_val = (AD**2 + ab_len**2 - BD**2) / (2 * AD * ab_len)
        skew_rad = math.acos(np.clip(cos_val, -1.0, 1.0))
        skew_deg = 90.0 - math.degrees(skew_rad)

        print(f"[skew] Calculated skew: {skew_deg:.2f} degrees")
        print("\n=== Klipper Configuration ===")
        print("Run these commands in your Klipper console (e.g., Mainsail/Fluidd):")
        print(f"SET_SKEW XY={AC:.3f},{BD:.3f},{AD:.3f}")
        print("SKEW_PROFILE SAVE=my_camera_skew")
        print("SAVE_CONFIG")
        print("\nFor more info (how to load the profile): https://www.klipper3d.org/Skew_Correction.html")


def do_generate_board(args):
    """Action for 'generate-board' command."""
    board = CharucoBoard()
    pdf_gen = PDFGenerator(board)
    pdf_gen.generate_pdfs()


def do_capture(args):
    """Action for 'capture-calibration-images' command for intrinsics."""
    camera = Camera(args.camera_id)
    moonraker = MoonrakerController(args.moonraker_url)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"[info] Saving images to: {out_dir}")

    (xmin, ymin), (xmax, ymax) = moonraker.get_bed_size(margin=args.margin)
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    offsets = [
        (0, 0), (args.dx, 0), (-args.dx, 0), (0, args.dy), (0, -args.dy),
        (args.dx, args.dy), (args.dx, -args.dy), (-args.dx, args.dy), (-args.dx, -args.dy),
    ]

    def perform_capture_sequence(phase_name: str):
        for i, dz in enumerate([0.0, 10.0, 20.0, 45.0]):
            z = args.z + dz
            print(f"\n--- {phase_name}, height Z={z:.1f} ---")
            for j, (ox, oy) in enumerate(offsets):
                tx = np.clip(cx + ox, xmin, xmax)
                ty = np.clip(cy + oy, ymin, ymax)
                moonraker.move_head(tx, ty, z)
                time.sleep(DWELL_AFTER_MOVE_SEC)
                img = camera.capture_frame()
                if img is not None:
                    fname = f"{phase_name}_z{z:.0f}_p{j:02d}.jpg"
                    path = os.path.join(out_dir, fname)
                    cv.imwrite(path, img)
                    print(f"  Saved {path}")
                else:
                    print(f"  [error] Failed to capture frame at ({tx:.1f}, {ty:.1f})")

    perform_capture_sequence("neutral")
    for phase in ("tilt1", "tilt2"):
        try:
            input(f"\n[action] Please tilt the bed slightly for perspective diversity ({phase}). Press Enter to continue...")
        except EOFError:
            print("\n[info] Non-interactive mode detected, continuing...")
        perform_capture_sequence(phase)

    print(f"\n[ok] Capture complete. Calibrate with:\n  python3 {sys.argv[0]} calibrate-camera \"{out_dir}/*.jpg\"")


def do_calibrate_camera(args):
    """Action for 'calibrate-camera' command."""
    board = CharucoBoard()
    calibrator = IntrinsicsCalibrator(board)
    calibrator.calibrate_from_images(args.images_glob)


def do_skew(args):
    """Action for 'skew' command."""
    camera = Camera(args.camera_id)
    moonraker = MoonrakerController(args.moonraker_url)
    board = CharucoBoard()
    calibrator = SkewCalibrator(camera, moonraker, board)
    calibrator.run_skew_calibration(args.z, args.margin)


def main():
    """Main function to parse arguments and dispatch commands."""
    if not hasattr(cv.aruco, "CharucoDetector"):
        print(
            f"[fatal] OpenCV {cv.__version__} lacks cv.aruco.CharucoDetector. "
            "Please install the 'opencv-contrib-python' package.",
            file=sys.stderr,
        )
        sys.exit(1)

    default_printer = os.environ.get("PRINTER_URL", "http://klipper.local")
    default_moonraker = f"{default_printer}:7125"
    default_camera = f"{default_printer}/webcam/?action=snapshot"

    parser = argparse.ArgumentParser(
        description="A comprehensive toolkit for Klipper XY skew calibration.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True, help="Available commands")

    # --- generate-board ---
    subparsers.add_parser(
        "generate-board",
        help="Generate the ChArUco board and save it as A4 and Letter PDFs."
    )

    # --- capture-calibration-images ---
    p_cap = subparsers.add_parser(
        "capture-calibration-images",
        help="Run a guided capture sequence for camera intrinsics calibration."
    )
    p_cap.add_argument("--camera-id", default=default_camera, help="Camera URL.")
    p_cap.add_argument("--out-dir", default="camera_calibration_imgs", help="Base directory to save images.")
    p_cap.add_argument("--z", type=float, default=40.0, help="Starting Z height (mm).")
    p_cap.add_argument("--margin", type=float, default=10.0, help="Bed safety margin (mm).")
    p_cap.add_argument("--dx", type=float, default=20.0, help="X offset for capture ring (mm).")
    p_cap.add_argument("--dy", type=float, default=20.0, help="Y offset for capture ring (mm).")
    p_cap.add_argument("--moonraker-url", default=default_moonraker, help="Moonraker base URL.")


    # --- calibrate-camera ---
    p_cal = subparsers.add_parser(
        "calibrate-camera",
        help="Calibrate camera intrinsics from captured images."
    )
    p_cal.add_argument("images_glob", help="Glob pattern for images, e.g., 'calibration_imgs/*.jpg'")

    # --- skew ---
    p_skew = subparsers.add_parser(
        "skew",
        help="Automatically calculate Klipper XY skew parameters."
    )
    p_skew.add_argument("--camera-id", default=default_camera, help="Camera URL or index.")
    p_skew.add_argument("--z", type=float, default=40.0, help="Z height for skew capture (mm).")
    p_skew.add_argument("--margin", type=float, default=10.0, help="Bed safety margin (mm).")
    p_skew.add_argument("--moonraker-url", default=default_moonraker, help="Moonraker base URL.")

    args = parser.parse_args()

    try:
        if args.cmd == "generate-board":
            do_generate_board(args)
        elif args.cmd == "capture-calibration-images":
            do_capture(args)
        elif args.cmd == "calibrate-camera":
            do_calibrate_camera(args)
        elif args.cmd == "skew":
            do_skew(args)
    except (RuntimeError, FileNotFoundError, requests.RequestException, ValueError) as e:
        print(f"\n[fatal] An error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()