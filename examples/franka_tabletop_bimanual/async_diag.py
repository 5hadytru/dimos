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

"""Diagnose why async position control stopped following targets after settling.

Same wrist move as live_test.py's `wrist` stage (joint 7 out ~6 deg, hold 1 s, back,
hold 1 s), driven the way Franka's async_position_control example drives it: every
cycle calls get_target_feedback() (which reads state through the control session),
then sets the target. No separate robot.read_once() while the session runs.

Safety: the session starts by holding the current pose and is ended by cancelling
(dropping the handler), never stop_control(). There is NO in-loop drift guard here
(that would need the extra reads this test leaves out): keep a hand on the e-stop.
The script reports at the end how far every joint moved.

    ~/dimos-franka/.venv/bin/python async_diag.py [--ip 10.42.0.230]
"""

import argparse
import math
import time

import pylibfranka as franka

TOL = math.radians(1.0)  # reported, not enforced
SPEEDS = [v * 0.25 for v in (2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26)]

parser = argparse.ArgumentParser()
parser.add_argument("--ip", default="10.42.0.230")
args = parser.parse_args()

robot = franka.Robot(args.ip, franka.RealtimeConfig.kIgnore)
start = list(robot.read_once().q)
print("Start (deg):", [round(math.degrees(q), 1) for q in start])
q7 = start[6]
path = (
    [q7 + 0.001 * i for i in range(101)]
    + [q7 + 0.1] * 100
    + [q7 + 0.1 - 0.001 * i for i in range(101)]
    + [q7] * 100
)

result = franka.AsyncPositionControlHandler.configure(
    robot,
    franka.AsyncPositionControlHandler.Configuration(
        maximum_joint_velocities=SPEEDS, goal_tolerance=0.01
    ),
)
if result.error_message is not None:
    raise SystemExit(f"configure failed: {result.error_message}")
handler = result.handler
del result


def target(q: list[float]) -> None:
    r = handler.set_joint_position_target(
        franka.AsyncPositionControlHandler.JointPositionTarget(joint_positions=q)
    )
    if r.error_message is not None:
        raise RuntimeError(f"set target failed: {r.error_message}")


try:
    target(start)  # hold first
    lo, hi = q7, q7
    for i, t7 in enumerate(path):
        fb = handler.get_target_feedback()
        if fb.error_message is not None:
            raise RuntimeError(f"feedback: {fb.error_message}")
        lo, hi = min(lo, t7), max(hi, t7)
        target(start[:6] + [t7])
        if i % 25 == 0:
            print(f"  i={i:3d} target7={math.degrees(t7):6.1f} deg  feedback={fb.status.name}")
        time.sleep(0.01)
    time.sleep(0.5)
    fb = handler.get_target_feedback()
    print("Final feedback:", fb.status.name, fb.error_message or "")
finally:
    del handler  # cancel the session (StopMove), never stop_control()
    time.sleep(0.5)
    end = list(robot.read_once().q)
    print("End (deg):  ", [round(math.degrees(q), 1) for q in end])
    others = max(abs(a - b) for a, b in zip(end[:6], start[:6]))
    print(
        f"joint 7: {math.degrees(start[6]):.1f} -> {math.degrees(end[6]):.1f} deg "
        f"(should end where it started); other joints moved {math.degrees(others):.2f} deg"
    )
    if others > TOL:
        print("WARNING: other joints moved more than 1 deg")
