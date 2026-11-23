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

"""Read-only FCI check: connect to the FR3 and print its state. Sends no motion commands."""

import argparse
import math

import pylibfranka as franka

parser = argparse.ArgumentParser()
parser.add_argument("--ip", default="10.42.0.230")
args = parser.parse_args()

robot = franka.Robot(args.ip, franka.RealtimeConfig.kIgnore)
state = robot.read_once()

print("Robot mode:", state.robot_mode)
print("Joint angles (deg):", [round(math.degrees(q), 1) for q in state.q])
print("End-effector position (m):", [round(x, 3) for x in list(state.O_T_EE)[12:15]])

gripper = franka.Gripper(args.ip)
g = gripper.read_once()
print(f"Gripper width: {g.width * 1000:.1f} mm (max {g.max_width * 1000:.1f} mm)")
