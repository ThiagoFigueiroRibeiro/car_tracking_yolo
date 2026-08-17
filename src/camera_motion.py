import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any


class CameraMotionEstimator:
    """
    Estimates camera motion between consecutive frames and maintains a cumulative
    transform that maps current-frame points into a stable reference frame.

    Approach:
    - ORB feature matching
    - RANSAC homography
    - fallback to translation if homography fails
    """

    def __init__(
        self,
        max_features: int = 1500,
        min_matches: int = 12,
        ratio_test: float = 0.75,
    ):
        self.orb = cv2.ORB_create(max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.min_matches = min_matches
        self.ratio_test = ratio_test

        self.prev_gray: Optional[np.ndarray] = None

        # cumulative warp: current frame -> reference frame
        self.global_warp = np.eye(3, dtype=np.float32)

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _translation_matrix(dx: float, dy: float) -> np.ndarray:
        H = np.eye(3, dtype=np.float32)
        H[0, 2] = dx
        H[1, 2] = dy
        return H

    def update(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Update motion estimate using the current frame.

        Returns:
            global_warp: 3x3 matrix mapping current-frame points into the reference frame
            info: debug dictionary
        """
        gray = self._to_gray(frame)

        if self.prev_gray is None:
            self.prev_gray = gray
            return self.global_warp.copy(), {
                "initialized": True,
                "method": "identity",
                "matches": 0,
                "inliers": 0,
            }

        kp_prev, des_prev = self.orb.detectAndCompute(self.prev_gray, None)
        kp_curr, des_curr = self.orb.detectAndCompute(gray, None)

        if des_prev is None or des_curr is None or len(kp_prev) < 4 or len(kp_curr) < 4:
            self.prev_gray = gray
            return self.global_warp.copy(), {
                "initialized": False,
                "method": "identity_no_descriptors",
                "matches": 0,
                "inliers": 0,
            }

        # Match current -> previous so we can estimate H(curr -> prev)
        knn = self.matcher.knnMatch(des_curr, des_prev, k=2)

        good = []
        for m, n in knn:
            if m.distance < self.ratio_test * n.distance:
                good.append(m)

        if len(good) < self.min_matches:
            self.prev_gray = gray
            return self.global_warp.copy(), {
                "initialized": False,
                "method": "identity_few_matches",
                "matches": len(good),
                "inliers": 0,
            }

        curr_pts = np.float32([kp_curr[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        prev_pts = np.float32([kp_prev[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Homography current -> previous
        H_curr_to_prev, mask = cv2.findHomography(
            curr_pts,
            prev_pts,
            cv2.RANSAC,
            5.0,
        )

        method = "homography"
        inliers = int(mask.sum()) if mask is not None else 0

        if H_curr_to_prev is None:
            # Fallback: median translation from matched points
            curr_flat = curr_pts.reshape(-1, 2)
            prev_flat = prev_pts.reshape(-1, 2)
            d = prev_flat - curr_flat
            dx = float(np.median(d[:, 0]))
            dy = float(np.median(d[:, 1]))
            H_curr_to_prev = self._translation_matrix(dx, dy)
            method = "translation_fallback"
            inliers = 0

        # Compose cumulative warp:
        # previous global_warp maps prev -> reference
        # current global_warp maps curr -> reference
        # current->reference = (prev->reference) @ (curr->prev)
        self.global_warp = self.global_warp @ H_curr_to_prev

        self.prev_gray = gray

        return self.global_warp.copy(), {
            "initialized": False,
            "method": method,
            "matches": len(good),
            "inliers": inliers,
        }

    @staticmethod
    def transform_point(point: np.ndarray, H: np.ndarray) -> np.ndarray:
        """
        Apply a 3x3 homography to a 2D point.
        """
        x, y = float(point[0]), float(point[1])
        p = np.array([x, y, 1.0], dtype=np.float32)
        q = H @ p
        if abs(q[2]) < 1e-6:
            return np.array([x, y], dtype=np.float32)
        return np.array([q[0] / q[2], q[1] / q[2]], dtype=np.float32)

    @classmethod
    def transform_points(cls, points: Dict[int, np.ndarray], H: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Transform a dict of {track_id: point} using the homography.
        """
        out = {}
        for track_id, pt in points.items():
            out[track_id] = cls.transform_point(pt, H)
        return out