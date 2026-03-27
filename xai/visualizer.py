"""
XAI side panel: renders DeepLIFT attribution results as Panda3D OnscreenImages.
Updates every N frames via background worker. Non-blocking.
"""
import logging
from typing import Optional

import numpy as np

from xai.deeplift import DeepLIFTWorker, AttributionResult, OUTPUT_NAMES

log = logging.getLogger(__name__)


def _numpy_to_panda_texture(arr: np.ndarray):
    """Convert (H, W, 3) uint8 RGB numpy array to a Panda3D Texture."""
    from panda3d.core import Texture, PNMImage
    h, w = arr.shape[:2]
    img = PNMImage(w, h, 3)
    for y in range(h):
        for x in range(w):
            r, g, b = arr[y, x]
            img.set_xel_val(x, y, r, g, b)
    tex = Texture()
    tex.load(img)
    return tex


class XAIPanel:
    PANEL_SCALE = 0.28   # size of each image panel in aspect2d units
    PANEL_X = 1.15       # right side of screen

    def __init__(self, base, worker: DeepLIFTWorker, update_interval: int = 5):
        self.base = base
        self.worker = worker
        self.update_interval = update_interval
        self._frame_count = 0
        self.xai_output_index = 0
        self._last_result: Optional[AttributionResult] = None
        self._setup_panels()

    def _setup_panels(self) -> None:
        from direct.gui.OnscreenImage import OnscreenImage
        from direct.gui.OnscreenText import OnscreenText

        a2d = self.base.aspect2d
        s = self.PANEL_SCALE
        x = self.PANEL_X

        # Three stacked panels: original, heatmap, overlay
        self._panels = []
        y_positions = [0.65, 0.1, -0.45]
        labels_text = ["Input Frame", "Attribution", "Overlay"]

        for i, (y, lbl) in enumerate(zip(y_positions, labels_text)):
            # Placeholder black image
            blank = np.zeros((224, 224, 3), dtype=np.uint8)
            tex = _numpy_to_panda_texture(blank)
            img = OnscreenImage(image=tex, pos=(x, 0, y), scale=s, parent=a2d)
            self._panels.append(img)
            OnscreenText(
                text=lbl, pos=(x, y - s - 0.04), scale=0.045,
                fg=(1, 1, 1, 1), shadow=(0, 0, 0, 0.8),
                mayChange=False, parent=a2d,
            )

        # Output name and haze label
        self._output_lbl = OnscreenText(
            text=f"XAI: {OUTPUT_NAMES[0]}", pos=(x, 0.95), scale=0.055,
            fg=(1, 0.9, 0.3, 1), shadow=(0, 0, 0, 0.8),
            mayChange=True, parent=a2d,
        )
        self._haze_lbl = OnscreenText(
            text="NON-HAZE", pos=(x, 0.88), scale=0.045,
            fg=(0.6, 0.9, 1, 1), shadow=(0, 0, 0, 0.8),
            mayChange=True, parent=a2d,
        )

        # Attribution legend
        OnscreenText(
            text="■ High  ■ Low", pos=(x, -0.82), scale=0.04,
            fg=(1, 0.4, 0.1, 1), shadow=(0, 0, 0, 0.8),
            mayChange=False, parent=a2d,
        )

    def maybe_update(self, frame_tensor, haze_active: bool) -> None:
        """Submit XAI job every N frames; refresh display if new result ready."""
        self._frame_count += 1

        if self._frame_count % self.update_interval == 0:
            self.worker.submit(frame_tensor, self.xai_output_index, haze_active)

        result = self.worker.get_latest()
        if result is not None and result is not self._last_result:
            self._last_result = result
            self._refresh_display(result)

    def _refresh_display(self, result: AttributionResult) -> None:
        images = [result.original, result.heatmap, result.overlay]
        for panel, arr in zip(self._panels, images):
            tex = _numpy_to_panda_texture(arr)
            panel.set_texture(tex, 1)

        self._output_lbl.setText(f"XAI: {result.output_name}")
        self._haze_lbl.setText(result.haze_label)

    def cycle_output(self) -> None:
        self.xai_output_index = (self.xai_output_index + 1) % 4
        self._output_lbl.setText(f"XAI: {OUTPUT_NAMES[self.xai_output_index]}")

    def switch_model(self, model) -> None:
        self.worker.switch_model(model)
