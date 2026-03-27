"""
Haze effect: applies white overlay to frames (matching training augmentation)
and Panda3D exponential fog to the 3D scene.
"""
import numpy as np
import cv2


class HazeEffect:
    def __init__(self, intensity: float = 0.6):
        self.intensity = intensity
        self._fog = None

    # ------------------------------------------------------------------
    # Frame-level haze (for inference preprocessing)
    # ------------------------------------------------------------------

    def apply_to_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Blend frame with white using cv2.addWeighted.
        Matches training augmentation: addWeighted(img, 1-alpha, white, alpha, 0).
        """
        white = np.full_like(frame, 255)
        return cv2.addWeighted(frame, 1.0 - self.intensity, white, self.intensity, 0)

    # ------------------------------------------------------------------
    # Scene-level fog (Panda3D)
    # ------------------------------------------------------------------

    def apply_to_scene(self, render) -> None:
        """Attach exponential fog to the Panda3D render node."""
        try:
            from panda3d.core import Fog
            if self._fog is None:
                self._fog = Fog("haze_fog")
                self._fog.set_mode(Fog.MExponential)
                self._fog.set_color(0.9, 0.9, 0.9, 1.0)
                self._fog.set_density(0.04)
            render.set_fog(self._fog)
        except ImportError:
            pass  # Panda3D not available in test context

    def remove_from_scene(self, render) -> None:
        """Remove fog from the Panda3D render node."""
        try:
            render.clear_fog()
        except Exception:
            pass
