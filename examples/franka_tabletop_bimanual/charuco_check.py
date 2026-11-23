# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Detect the ChArUco board in one wrist-camera frame (no robot motion).

Board: 8x5 squares, 30 mm squares, 22 mm markers, DICT_APRILTAG_36h11.
Prints how many corners were found and the board's pose in the colour camera's
optical frame, and saves an annotated image next to this script.

    ~/dimos-franka/.venv/bin/python charuco_check.py [--square 0.030] [--marker 0.022]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

parser = argparse.ArgumentParser()
parser.add_argument("--square", type=float, default=0.030)
parser.add_argument("--marker", type=float, default=0.022)
parser.add_argument("--out", default=str(Path(__file__).with_name("charuco_check.png")))
args = parser.parse_args()

board = cv2.aruco.CharucoBoard(
    (8, 5),
    args.square,
    args.marker,
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
)
detector = cv2.aruco.CharucoDetector(board)

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
prof = pipe.start(cfg)
try:
    for _ in range(30):
        frame = pipe.wait_for_frames(5000)
    color = frame.get_color_frame()
    intr = color.profile.as_video_stream_profile().intrinsics
    img = np.asanyarray(color.get_data()).copy()
finally:
    pipe.stop()

K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
dist = np.array(intr.coeffs[:5])
corners, ids, marker_corners, marker_ids = detector.detectBoard(img)
n_markers = 0 if marker_ids is None else len(marker_ids)
n_corners = 0 if ids is None else len(ids)
print(f"markers found: {n_markers}/20, chessboard corners found: {n_corners}/28")

if n_corners >= 6:
    obj, imgp = board.matchImagePoints(corners, ids)
    ok, rvec, tvec = cv2.solvePnP(obj, imgp, K, dist)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - imgp.reshape(-1, 2)) ** 2, axis=1)))
    R, _ = cv2.Rodrigues(rvec)
    tilt = np.degrees(np.arccos(abs(R[2, 2])))
    print(
        f"board origin in camera (m): {tvec.ravel().round(3)}  distance {np.linalg.norm(tvec):.3f} m"
    )
    print(f"board tilt vs camera axis: {tilt:.1f} deg, reprojection error {err:.2f} px")
    cv2.aruco.drawDetectedCornersCharuco(img, corners, ids)
    cv2.drawFrameAxes(img, K, dist, rvec, tvec, 0.06)
elif n_markers:
    cv2.aruco.drawDetectedMarkers(img, marker_corners, marker_ids)
cv2.imwrite(args.out, img)
print(f"saved {args.out}")
