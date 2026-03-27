"""
HUD overlay rendered on Panda3D's aspect2d layer.
Shows: steering arc, gas/brake bars, gear, speed, frame counter,
haze status, model label, pause indicator.
"""
import math
import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class SimState:
    paused: bool = False
    haze_active: bool = False
    active_model: str = "cnn"
    manual_override: bool = False
    xai_output_index: int = 0
    frame_counter: int = 0


class HUDRenderer:
    # Steering arc range from dataset
    STEER_MIN = -122.0
    STEER_MAX = 115.0

    def __init__(self, base):
        self.base = base
        self._nodes = []
        self._setup()

    def _setup(self) -> None:
        from direct.gui.DirectGui import DirectWaitBar
        from panda3d.core import TextNode, NodePath
        from direct.gui.OnscreenText import OnscreenText

        a2d = self.base.aspect2d

        # --- Steering arc (LineSegs drawn dynamically) ---
        from panda3d.core import LineSegs
        self._steer_segs = LineSegs("steer_arc")
        self._steer_np = a2d.attach_new_node(self._steer_segs.create())
        self._steer_np.set_pos(-1.1, 0, -0.6)
        self._steer_np.set_scale(0.25)

        # --- Gas bar ---
        self._gas_bar = DirectWaitBar(
            text="", value=0, range=1,
            pos=(-1.55, 0, -0.5), scale=(0.04, 1, 0.35),
            frameColor=(0.1, 0.1, 0.1, 0.8),
            barColor=(0.1, 0.9, 0.1, 1),
            parent=a2d,
        )

        # --- Brake bar ---
        self._brake_bar = DirectWaitBar(
            text="", value=0, range=1,
            pos=(-1.45, 0, -0.5), scale=(0.04, 1, 0.35),
            frameColor=(0.1, 0.1, 0.1, 0.8),
            barColor=(0.9, 0.1, 0.1, 1),
            parent=a2d,
        )

        def _label(text, pos, scale=0.055, fg=(1,1,1,1), align="left"):
            return OnscreenText(
                text=text, pos=pos, scale=scale, fg=fg,
                shadow=(0,0,0,0.8), shadowOffset=(0.05,0.05),
                mayChange=True, parent=a2d,
            )

        self._gear_lbl   = _label("G:0",  (-1.62, 0, -0.88))
        self._speed_lbl  = _label("0 km/h", (-1.62, 0, -0.78))
        self._frame_lbl  = _label("F:0",  (-1.62, 0, -0.68))
        self._haze_lbl   = _label("CLEAR", (-1.62, 0, -0.58), fg=(0.6,0.9,1,1))
        self._model_lbl  = _label("CNN",  (-1.62, 0, -0.48), fg=(1,0.9,0.3,1))
        self._steer_lbl  = _label("0°",   (-1.05, 0, -0.88))
        self._pause_lbl  = OnscreenText(
            text="PAUSED", pos=(0, 0.1), scale=0.12,
            fg=(1, 0.3, 0.3, 1), shadow=(0,0,0,0.9),
            mayChange=True, parent=a2d,
        )
        self._pause_lbl.hide()

        # Bar labels
        _label("GAS",   (-1.55, 0, -0.88), scale=0.04)
        _label("BRK",   (-1.45, 0, -0.88), scale=0.04)

    def update(self, vehicle_state, output, sim: SimState) -> None:
        """Update all HUD elements. Called every frame."""
        # Gas / brake bars
        self._gas_bar["value"]   = output.gas_pedal
        self._brake_bar["value"] = output.brake_pedal

        # Labels
        self._gear_lbl.setText(f"G:{output.gear}")
        self._speed_lbl.setText(f"{vehicle_state.speed:.0f} km/h")
        self._frame_lbl.setText(f"F:{sim.frame_counter}")
        self._haze_lbl.setText("HAZY" if sim.haze_active else "CLEAR")

        if sim.manual_override:
            self._model_lbl.setText("MANUAL")
        else:
            self._model_lbl.setText(sim.active_model.upper())

        self._steer_lbl.setText(f"{output.steering_degrees:.0f}°")

        # Pause indicator
        if sim.paused:
            self._pause_lbl.show()
        else:
            self._pause_lbl.hide()

        # Redraw steering arc
        self._draw_steering_arc(output.steering_degrees)

    def _draw_steering_arc(self, angle_deg: float) -> None:
        """Draw arc background and needle for current steering angle."""
        from panda3d.core import LineSegs
        ls = LineSegs("steer_arc")
        ls.set_thickness(2)

        # Background arc
        ls.set_color(0.4, 0.4, 0.4, 1)
        steps = 40
        total_range = self.STEER_MAX - self.STEER_MIN
        for i in range(steps):
            a = math.radians(self.STEER_MIN + (i / steps) * total_range - 90)
            x, z = math.cos(a), math.sin(a)
            if i == 0:
                ls.move_to(x, 0, z)
            else:
                ls.draw_to(x, 0, z)

        # Needle
        ls.set_color(1, 0.8, 0, 1)
        ls.set_thickness(3)
        needle_angle = math.radians(angle_deg - 90)
        ls.move_to(0, 0, 0)
        ls.draw_to(math.cos(needle_angle) * 0.9, 0, math.sin(needle_angle) * 0.9)

        # Replace node
        self._steer_np.remove_node()
        self._steer_np = self.base.aspect2d.attach_new_node(ls.create())
        self._steer_np.set_pos(-1.1, 0, -0.6)
        self._steer_np.set_scale(0.25)
