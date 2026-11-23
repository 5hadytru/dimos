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

"""Repeat: find the blue block, pick it, place it a few cm away, go home, wait 5 s.

Runs against a running `dimos run fr3-grasp ...` stack until Ctrl+C. Any step
that does not succeed stops the loop (the arm is left where that step left it).
A block found outside the work area below is left alone: the loop waits for it
to be moved back in.

    ~/dimos-franka/.venv/bin/python pick_place_loop.py
"""

import math
import random
import re
import time

from dimos.porcelain.dimos import Dimos

LABEL = "blue block"
# Where the block may be picked from and placed: on the table, in the wrist
# camera's view from the ready pose, clear of the table's far edge.
X_RANGE = (0.27, 0.37)
Y_RANGE = (-0.20, 0.26)
# Places must also be table, as the wrist camera measured it from the ready pose,
# for at least this far all around (m). The table edge is not square to the arm.
TABLE_MARGIN = 0.05
CELL = 0.02
# Distance from the base axis, m. Near the base the arm folds tightly; with the
# connection's tool contact thresholds at 20/40 N (its estimate read up to 12.6 N
# there with no contact) that is fine, so only the far limit is binding.
R_RANGE = (0.28, 0.45)
PLACE_STEP = (0.03, 0.06)  # how far each place is from the pick, m
# Tool point height when releasing: the pick's grasp height (0.023 m) plus 2 mm.
PLACE_Z = 0.025
WAIT_S = 5.0
MAX_REJECTS = 3  # looks in a row with every grasp rejected before giving up

FOUND = re.compile(
    r"'([^']+)' at \(([-\d.]+), ([-\d.]+), ([-\d.]+)\) in \w+ score=([\d.]+).*?id=(\w+)"
)


def inside(x: float, y: float, margin: float = 0.0) -> bool:
    r = math.hypot(x, y)
    return (
        X_RANGE[0] + margin <= x <= X_RANGE[1] - margin
        and Y_RANGE[0] + margin <= y <= Y_RANGE[1] - margin
        and R_RANGE[0] + margin <= r <= R_RANGE[1] - margin
    )


def measure_table(app: Dimos) -> set[tuple[int, int]]:
    """The 2 cm cells (x, y) that are table, from one wrist-camera depth image at the ready pose."""
    import json

    import numpy as np
    from scipy.spatial.transform import Rotation

    from handeye import Model, tf

    cal = json.loads(open("handeye_refined.json").read())
    T_hand_link = tf(Rotation.from_quat(cal["rotation_xyzw"]).as_matrix(), cal["translation_m"])
    T_link_opt = np.load("handeye_data.npz")["T_link_opt"]
    group = app.ManipulationModule.get_state().groups["manipulator"]
    T_base_opt = Model().hand(list(group.joints.position)) @ T_hand_link @ T_link_opt
    loc = app.ManipulationLocalize
    info = loc.peek_stream("camera_info", 5.0)
    depth = np.asarray(loc.peek_stream("depth_image", 5.0).data).astype(float) / 1000.0
    K = np.asarray(info.K).reshape(3, 3)
    v, u = np.nonzero(depth > 0)
    z = depth[v, u]
    pts = np.c_[(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z, np.ones_like(z)]
    base = (T_base_opt @ pts.T).T
    on_table = (base[:, 2] > -0.02) & (base[:, 2] < 0.05)  # the top, the paper, the block
    cells = {(int(round(px / CELL)), int(round(py / CELL))) for px, py in base[on_table, :2]}
    return cells


def on_table(cells: set[tuple[int, int]], x: float, y: float) -> bool:
    r = int(math.ceil(TABLE_MARGIN / CELL))
    cx, cy = int(round(x / CELL)), int(round(y / CELL))
    return all(
        (cx + i, cy + j) in cells
        for i in range(-r, r + 1)
        for j in range(-r, r + 1)
        if i * i + j * j <= r * r
    )


def next_spot(x: float, y: float, cells: set[tuple[int, int]]) -> tuple[float, float]:
    for _ in range(500):
        d = random.uniform(*PLACE_STEP)
        a = random.uniform(0, 2 * math.pi)
        nx, ny = x + d * math.cos(a), y + d * math.sin(a)
        if inside(nx, ny, margin=0.01) and on_table(cells, nx, ny):
            return nx, ny
    raise RuntimeError("no place spot inside the work area and on the table")


def ok(result: object, word: str) -> bool:
    msg = str(getattr(result, "message", result))
    if msg.startswith(("Execution ended with", "No plan")):
        return False
    return word.lower() in msg.lower()


def at_home(app: Dimos, tol_deg: float = 2.0) -> bool:
    """Whether the arm's measured joints are within tol_deg of the home preset."""
    group = app.ManipulationModule.get_state().groups["manipulator"]
    home = list(group.joint_presets["home"].position)
    if group.joints is None:
        print("  no joint reading")
        return False
    now = list(group.joints.position)
    worst = max(abs(a - b) for a, b in zip(now, home))
    if math.degrees(worst) > tol_deg:
        print(f"  not at home: a joint is {math.degrees(worst):.1f} deg off")
        return False
    return True


def go_home(app: Dimos, skills) -> bool:
    r = skills.go_home()
    print(f"  home: {getattr(r, 'message', r)[:80]}")
    time.sleep(1.0)  # let the arm settle and the reading catch up
    return ok(r, "complet") and at_home(app)


def main() -> None:
    app = Dimos.connect()
    loc, pnp, skills = app.ManipulationLocalize, app.PickAndPlaceModule, app.ManipulationSkills
    n = 0
    rejected = 0
    try:
        if not go_home(app, skills):
            print("did not reach the ready pose; stopping")
            return
        table = measure_table(app)
        print(f"table measured: {len(table)} cells of {CELL * 100:.0f} cm")
        while True:
            time.sleep(WAIT_S)
            answer = loc.localize(LABEL, -4.0, 4.0)
            m = FOUND.search(answer)
            if not m:
                print(f"[{n}] not found, looking again: {answer.splitlines()[-1]}")
                continue
            x, y, z, score, oid = float(m[2]), float(m[3]), float(m[4]), float(m[5]), m[6]
            if not inside(x, y):
                print(
                    f"[{n}] block at ({x:.3f}, {y:.3f}) is outside the work area "
                    f"(x {X_RANGE[0]}-{X_RANGE[1]}, y {Y_RANGE[0]}-{Y_RANGE[1]}, "
                    f"{R_RANGE[0]}-{R_RANGE[1]} m from the base, now {math.hypot(x, y):.2f}); "
                    "move it back in, looking again"
                )
                continue
            print(f"[{n}] block at ({x:.3f}, {y:.3f}, {z:.3f}) score {score:.2f}; picking")
            r = pnp.pick_object(oid)
            print(f"[{n}] pick: {r}")
            if "Generated 0 grasp candidates" in str(r):
                # Nothing moved: the contact check rejected every grasp, often from
                # a noisy single view of the block. Look again with fresh frames.
                rejected += 1
                if rejected >= MAX_REJECTS:
                    print(f"[{n}] no usable grasp in {MAX_REJECTS} looks; stopping")
                    return
                print(
                    f"[{n}] no grasp passed the contact check; looking again ({rejected}/{MAX_REJECTS})"
                )
                continue
            if not ok(r, "pick complete"):
                return
            rejected = 0
            px, py = next_spot(x, y, table)
            print(
                f"[{n}] placing at ({px:.3f}, {py:.3f}), {math.hypot(px - x, py - y) * 100:.1f} cm away"
            )
            r = pnp.place_at(px, py, PLACE_Z)
            print(f"[{n}] place: {r}")
            if not ok(r, "place complete"):
                return
            if not go_home(app, skills):
                print(f"[{n}] did not reach the ready pose; stopping")
                return
            n += 1
            print(f"--- {n} cycle(s) done; next in {WAIT_S:.0f} s ---")
    except KeyboardInterrupt:
        print(f"\nstopped by you after {n} cycle(s); cancelling any motion")
        try:
            skills.cancel()
        except Exception:
            pass
    finally:
        app.stop()


if __name__ == "__main__":
    main()
