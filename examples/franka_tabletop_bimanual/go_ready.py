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

"""Move the FR3 to Franka's ready pose through dimOS's FrankaConnection, slowly.

Streams a smooth joint-space path (every joint starts and stops together, cosine
speed profile) with the fastest joint at most MAX_SPEED rad/s. The connection's
guards stay on: hold-first session start, jump guard, drift guard, not-following
guard. Any fault stops the move.

    ~/dimos-franka/.venv/bin/python go_ready.py [--ip 10.42.0.230]
"""

import argparse
import math
import time

from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.robot.manipulators.franka.config import FR3_READY
from dimos.robot.manipulators.franka.connection import FrankaConnection
from live_test import JOINTS, LocalTransport, check, deg

MAX_SPEED = 0.15  # rad/s for the joint that moves furthest
PERIOD = 0.01


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="10.42.0.230")
    args = parser.parse_args()

    module = FrankaConnection(ip=args.ip, deadman_timeout_s=0.5)
    ports = {name: LocalTransport() for name in {**module.inputs, **module.outputs}}
    for name, transport in ports.items():
        module.set_transport(name, transport)  # type: ignore[arg-type]
    state, command = ports["joint_state"], ports["position_command"]

    def arm() -> list[float]:
        return list(state.latest[tuple(JOINTS)].position)

    module.start()
    try:
        time.sleep(0.5)
        check(module)
        start = arm()
        delta = [g - s for g, s in zip(FR3_READY, start)]
        biggest = max(abs(d) for d in delta)
        # Cosine profile peaks at pi/2 times the average speed.
        duration = max(2.0, biggest / MAX_SPEED * math.pi / 2)
        print("From (deg):", deg(start))
        print("To   (deg):", deg(FR3_READY))
        print(f"Largest change {math.degrees(biggest):.0f} deg; taking {duration:.1f} s")
        steps = int(duration / PERIOD)
        for i in range(steps + 1):
            s = 0.5 - 0.5 * math.cos(math.pi * i / steps)
            q = [a + s * d for a, d in zip(start, delta)]
            command.publish(JointState(name=list(JOINTS), position=q))
            time.sleep(PERIOD)
            if i % 100 == 0:
                check(module)
                print(f"  {100 * i / steps:3.0f}%  measured (deg): {deg(arm())}")
        for _ in range(100):  # hold the goal 1 s while it settles
            command.publish(JointState(name=list(JOINTS), position=list(FR3_READY)))
            time.sleep(PERIOD)
        check(module)
        end = arm()
        err = max(abs(a - b) for a, b in zip(end, FR3_READY))
        print("Done. Joints (deg):", deg(end), f"| largest error {math.degrees(err):.2f} deg")
    finally:
        module.stop()


if __name__ == "__main__":
    main()
