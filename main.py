"""
Self-Driving 3D Simulator
Entry point. Wires all subsystems together.

Controls:
  Space     - Pause / Resume
  H         - Toggle haze
  1         - Switch to CNN model
  2         - Switch to LTC model
  X         - Cycle XAI output (steering/gas/brake/gear)
  WASD/Arrows - Manual override (if enabled in config)
  ESC       - Quit
"""
import logging
import sys
import os

# Ensure simulator package root is on path
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

from direct.showbase.ShowBase import ShowBase
from panda3d.core import WindowProperties, ClockObject

from config import Config
from engine.scene import SceneManager
from engine.vehicle import VehicleController
from engine.haze import HazeEffect
from models.inference import InferencePipeline
from hud.dashboard import HUDRenderer, SimState
from xai.deeplift import DeepLIFTWorker
from xai.visualizer import XAIPanel


class SimulatorApp(ShowBase):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Window setup
        props = WindowProperties()
        props.set_title("Self-Driving Simulator | CNN vs LTC")
        props.set_size(config.window_width, config.window_height)
        self.win.request_properties(props)

        # Lock framerate
        ClockObject.get_global_clock().set_mode(ClockObject.MLimited)
        ClockObject.get_global_clock().set_frame_rate(config.target_fps)

        # Disable default mouse camera control
        self.disable_mouse()

        # Simulation state
        self.sim_state = SimState(
            haze_active=config.haze_active,
            active_model=config.active_model,
            manual_override=config.manual_override,
        )

        # Subsystems
        log.info("Initialising inference pipeline...")
        self.inference = InferencePipeline(config)

        log.info("Initialising haze effect...")
        self.haze_effect = HazeEffect(config.haze_intensity)

        log.info("Initialising scene...")
        self.scene = SceneManager(self, config)

        log.info("Initialising vehicle...")
        self.vehicle = VehicleController(
            max_speed=config.max_speed_kmh,
            accel_rate=config.accel_rate,
            decel_rate=config.decel_rate,
            turn_rate=config.turn_rate,
        )

        log.info("Initialising HUD...")
        self.hud = HUDRenderer(self)

        log.info("Initialising XAI worker...")
        self.xai_worker = DeepLIFTWorker(self.inference.get_active_model())
        self.xai_worker.start()
        self.xai_panel = XAIPanel(self, self.xai_worker, config.xai_update_interval)

        # Position main camera behind vehicle
        self.camera.set_pos(0, -12, 4)
        self.camera.look_at(0, 0, 1)

        # Apply initial haze state
        if config.haze_active:
            self.scene.set_haze(True, self.haze_effect)

        # Manual override state
        self._manual_steer = 0.0
        self._manual_throttle = 0.0

        self._setup_keys()
        self.task_mgr.add(self._main_loop, "main_loop")
        log.info("Simulator ready.")

    # ------------------------------------------------------------------
    # Key bindings
    # ------------------------------------------------------------------

    def _setup_keys(self) -> None:
        self.accept("space",       self.toggle_pause)
        self.accept("h",           self.toggle_haze)
        self.accept("1",           lambda: self.switch_model("cnn"))
        self.accept("2",           lambda: self.switch_model("ltc"))
        self.accept("x",           self.cycle_xai_output)
        self.accept("escape",      self.quit_sim)

        if self.config.manual_override:
            self.accept("arrow_left",  lambda: self._set_steer(-1.0))
            self.accept("arrow_right", lambda: self._set_steer(1.0))
            self.accept("arrow_left-up",  lambda: self._set_steer(0.0))
            self.accept("arrow_right-up", lambda: self._set_steer(0.0))
            self.accept("arrow_up",    lambda: self._set_throttle(1.0))
            self.accept("arrow_down",  lambda: self._set_throttle(-1.0))
            self.accept("arrow_up-up",   lambda: self._set_throttle(0.0))
            self.accept("arrow_down-up", lambda: self._set_throttle(0.0))
            self.accept("a", lambda: self._set_steer(-1.0))
            self.accept("d", lambda: self._set_steer(1.0))
            self.accept("a-up", lambda: self._set_steer(0.0))
            self.accept("d-up", lambda: self._set_steer(0.0))
            self.accept("w", lambda: self._set_throttle(1.0))
            self.accept("s", lambda: self._set_throttle(-1.0))
            self.accept("w-up", lambda: self._set_throttle(0.0))
            self.accept("s-up", lambda: self._set_throttle(0.0))

    def _set_steer(self, v: float) -> None:
        self._manual_steer = v

    def _set_throttle(self, v: float) -> None:
        self._manual_throttle = v

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _main_loop(self, task):
        from direct.task import Task
        dt = ClockObject.get_global_clock().get_dt()
        self.sim_state.frame_counter += 1

        if not self.sim_state.paused:
            # 1. Capture frame from scene camera
            frame = self.scene.capture_frame()

            # 2. Inference or manual
            if self.sim_state.manual_override:
                from models.inference import ModelOutput
                output = ModelOutput(
                    steering_degrees=self._manual_steer * 90.0,
                    gas_pedal=max(0.0, self._manual_throttle),
                    brake_pedal=max(0.0, -self._manual_throttle),
                    gear=self.vehicle.state.gear,
                )
            else:
                output = self.inference.run(frame, self.sim_state.haze_active)

            # 3. Vehicle physics
            self.vehicle.update(output, dt, paused=False)

            # 4. Scene tile recycling
            self.scene.update(self.vehicle.state.y)

            # 5. Follow camera
            self.camera.set_pos(
                self.vehicle.state.x,
                self.vehicle.state.y - 12,
                4,
            )
            self.camera.look_at(
                self.vehicle.state.x,
                self.vehicle.state.y,
                1,
            )

            # 6. XAI update (non-blocking)
            if not self.sim_state.manual_override:
                import torch
                frame_tensor = self.inference._preprocess(frame, self.sim_state.haze_active)
                self.xai_panel.maybe_update(frame_tensor, self.sim_state.haze_active)
        else:
            # Still need a dummy output for HUD when paused
            from models.inference import ModelOutput
            output = ModelOutput(0.0, 0.0, 0.0, self.vehicle.state.gear)

        # 7. HUD always updates
        self.hud.update(self.vehicle.state, output, self.sim_state)

        return Task.cont

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def toggle_pause(self) -> None:
        self.sim_state.paused = not self.sim_state.paused
        log.info("Simulation %s.", "paused" if self.sim_state.paused else "resumed")

    def toggle_haze(self) -> None:
        self.sim_state.haze_active = not self.sim_state.haze_active
        self.scene.set_haze(self.sim_state.haze_active, self.haze_effect)
        log.info("Haze %s.", "ON" if self.sim_state.haze_active else "OFF")

    def switch_model(self, key: str) -> None:
        success = self.inference.switch_model(key)
        if success:
            self.sim_state.active_model = key
            self.xai_panel.switch_model(self.inference.get_active_model())
            log.info("Switched to %s model.", key.upper())
        else:
            log.warning("Model '%s' not available.", key)

    def cycle_xai_output(self) -> None:
        self.xai_panel.cycle_output()
        self.sim_state.xai_output_index = self.xai_panel.xai_output_index

    def quit_sim(self) -> None:
        log.info("Shutting down...")
        self.xai_worker.stop()
        self.destroy()
        sys.exit(0)


def main():
    config = Config()
    app = SimulatorApp(config)
    app.run()


if __name__ == "__main__":
    main()
