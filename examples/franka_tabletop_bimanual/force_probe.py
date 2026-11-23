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

"""Was the cartesian_reflex at the folded pre-grasp pose a real contact or a false one?

Replays the joint pose where it stopped (read off the arm afterwards), from the
ready pose and back, first slowly, then at about the planner's speed, while
recording the robot's own estimate of the external force at the tool
(O_F_ext_hat_K) and of the external joint torques (tau_ext_hat_filtered).
The connection's collision thresholds are lower 10 N / upper 20 N at the tool.

    ~/dimos-franka/.venv/bin/python force_probe.py
"""

import math
import threading
import time

import numpy as np

import handeye
from handeye import Arm, Model

READY = [0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398]
STOP = [math.radians(d) for d in [-0.5, -7.8, 26.9, -170.2, 11.6, 163.1, 79.9]]
PASSES = [("slow", 0.15), ("planner speed", 0.5)]  # fastest joint, rad/s


def check_path(model: Model) -> None:
    """Joint-interpolated path ready -> stop: fingertips stay high, no self-collision."""
    lowest = 1.0
    for s in np.linspace(0, 1, 60):
        q = [a + s * (b - a) for a, b in zip(READY, STOP)]
        lowest = min(lowest, model.tcp(q)[2, 3])
        if not model.collision_free(q):
            raise SystemExit(f"self-collision along the path at s={s:.2f}; not moving")
    if lowest < 0.09:
        raise SystemExit(f"fingertips would dip to {lowest:.3f} m; not moving")
    print(f"path checked: fingertips stay at or above {lowest:.3f} m, no self-collision")


class Recorder:
    def __init__(self, arm: Arm) -> None:
        self.arm, self.rows, self.on = arm, [], True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while self.on:
            st = self.arm.module._state
            if st is not None:
                f = np.array(list(st.O_F_ext_hat_K)[:3])
                tau = np.array(list(st.tau_ext_hat_filtered))
                self.rows.append(
                    (time.monotonic(), np.linalg.norm(f), np.max(np.abs(tau)), st.robot_mode.name)
                )
            time.sleep(0.01)

    def window(self, t0: float, t1: float) -> tuple[float, float]:
        sel = [r for r in self.rows if t0 <= r[0] <= t1]
        return max(r[1] for r in sel), max(r[2] for r in sel)


def main() -> None:
    check_path(Model())
    arm = Arm()
    rec = Recorder(arm)
    try:
        if max(abs(a - b) for a, b in zip(arm.q(), READY)) > math.radians(1):
            raise SystemExit("not at the ready pose; send it home first")
        t = time.monotonic()
        arm.hold(2.0)
        f, tau = rec.window(t, time.monotonic())
        print(f"at rest, ready pose:      force {f:5.1f} N, joint torque {tau:4.1f} Nm")
        for name, speed in PASSES:
            handeye.MAX_SPEED = speed
            t = time.monotonic()
            arm.move(STOP)
            f, tau = rec.window(t, time.monotonic())
            print(f"{name:14s} out:        force {f:5.1f} N, joint torque {tau:4.1f} Nm")
            t = time.monotonic()
            arm.hold(3.0)
            f, tau = rec.window(t, time.monotonic())
            print(f"{name:14s} at the pose: force {f:5.1f} N, joint torque {tau:4.1f} Nm")
            t = time.monotonic()
            arm.move(READY)
            f, tau = rec.window(t, time.monotonic())
            print(f"{name:14s} back:       force {f:5.1f} N, joint torque {tau:4.1f} Nm")
    except SystemExit as e:
        print(e)
        reflexes = [r for r in rec.rows if r[3] == "Reflex"]
        if reflexes:
            print(f"reflex seen; peak force before it {max(r[1] for r in rec.rows):.1f} N")
    finally:
        rec.on = False
        handeye.MAX_SPEED = 0.15
        arm.close()


if __name__ == "__main__":
    main()
