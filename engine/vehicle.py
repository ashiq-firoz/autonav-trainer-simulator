"""
Vehicle state and kinematic physics controller.
Applies model outputs (steering, gas, brake, gear) to update position/heading/speed.
"""
import math
from dataclasses import dataclass, field
from typing import Optional

from models.inference import ModelOutput


@dataclass
class VehicleState:
    x: float = 0.0          # World X position (meters)
    y: float = 0.0          # World Y position (meters)
    heading: float = 0.0    # Heading in degrees (0 = +Y axis / north)
    speed: float = 0.0      # Current speed in km/h
    gear: int = 0


class VehicleController:
    def __init__(
        self,
        max_speed: float = 120.0,
        accel_rate: float = 20.0,
        decel_rate: float = 40.0,
        turn_rate: float = 60.0,
    ):
        self.max_speed = max_speed
        self.accel_rate = accel_rate    # km/h per second at full gas
        self.decel_rate = decel_rate    # km/h per second at full brake
        self.turn_rate = turn_rate      # degrees/second at full steer, full speed
        self.state = VehicleState()
        self.node = None                # Panda3D NodePath, set by SceneManager

    def update(self, output: ModelOutput, dt: float, paused: bool = False) -> None:
        """Apply model output to vehicle state. No-op when paused."""
        if paused:
            return

        s = self.state

        # Speed update
        s.speed += (output.gas_pedal * self.accel_rate
                    - output.brake_pedal * self.decel_rate) * dt
        s.speed = max(0.0, min(self.max_speed, s.speed))

        # Heading update — proportional to steer and speed
        max_steer = 122.0  # max abs steering degrees in dataset
        steer_norm = output.steering_degrees / max_steer   # [-1, 1]
        speed_factor = s.speed / self.max_speed
        s.heading += steer_norm * self.turn_rate * speed_factor * dt
        s.heading %= 360.0

        # Position update (convert km/h to m/s: /3.6)
        speed_ms = s.speed / 3.6
        rad = math.radians(s.heading)
        s.x += math.sin(rad) * speed_ms * dt
        s.y += math.cos(rad) * speed_ms * dt

        s.gear = output.gear

        # Sync Panda3D node if attached
        if self.node is not None:
            self.node.set_pos(s.x, s.y, 0)
            self.node.set_h(-s.heading)

    def apply_manual(self, steering: float, throttle: float, dt: float) -> None:
        """Manual override: steering in [-1,1], throttle in [-1,1]."""
        output = ModelOutput(
            steering_degrees=steering * 122.0,
            gas_pedal=max(0.0, throttle),
            brake_pedal=max(0.0, -throttle),
            gear=self.state.gear,
        )
        self.update(output, dt, paused=False)

    def reset(self) -> None:
        """Return vehicle to origin."""
        self.state = VehicleState()
        if self.node is not None:
            self.node.set_pos(0, 0, 0)
            self.node.set_h(0)
