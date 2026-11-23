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

"""Hardware check: with the Hand in its own process, do the arm's readings keep flowing
while the fingers move? Sends Hand targets only; the arm is not commanded.

    ~/dimos-franka/.venv/bin/python hand_process_test.py [ip] [source]
"""

import sys
import time

from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.robot.manipulators.franka.connection import FrankaConnection
from live_test import LocalTransport


def main() -> None:
    ip = sys.argv[1] if len(sys.argv) > 1 else "10.43.0.53"
    side = sys.argv[2] if len(sys.argv) > 2 else "left"
    m = FrankaConnection(ip=ip, source=side, deadman_timeout_s=5.0)
    ports = {n: LocalTransport() for n in {**m.inputs, **m.outputs}}
    for n, t in ports.items():
        m.set_transport(n, t)
    arrivals = []
    ports["joint_state"].subscribe(lambda msg: arrivals.append((time.monotonic(), list(msg.name))))
    m.start()
    try:
        time.sleep(1.0)
        for finger in (0.02, 0.04):  # per finger: 40 mm, then 80 mm across
            arrivals.clear()
            ports["position_command"].publish(
                JointState(name=[f"{side}/fr3_finger_joint1"], position=[finger])
            )
            time.sleep(2.5)
            arm = [t for t, n in arrivals if f"{side}/fr3_joint1" in n]
            gaps = [b - a for a, b in zip(arm, arm[1:])]
            print(
                f"Hand to {finger * 2000:.0f} mm: {len(arm)} arm readings in 2.5 s, longest gap "
                f"{max(gaps) * 1000:.0f} ms; Hand reads {m._width * 1000:.1f} mm"
            )
        print("status:", m.status().last_error)
    finally:
        m.stop()


if __name__ == "__main__":
    main()
