# Franka Research 3 Integration

dimOS drives a Franka Research 3 (FR3) with the Franka Hand through
`FrankaConnection` (`dimos/robot/manipulators/franka/`), over Franka's
pylibfranka and the Franka Control Interface (FCI).

## Requirements

- Robot system version 5.9 or newer, with the FCI feature installed (Desk →
  Settings → System → Installed Features). Install it from Franka World if it
  is missing.
- `pip install pylibfranka` (0.18 or newer; tested with 0.21.3 on system 5.10.0).
- The workstation on Control's LAN port (the shop floor network), not the port
  on the arm's base. Franka's default is a static `172.16.0.2` for Control and
  `172.16.0.1/24` for the workstation.

## Prepare the robot in Desk

1. Release the joint brakes and switch to **Execution** mode.
2. Open the menu under the robot's address in Desk's top bar and choose
   **Activate FCI**. Keep Desk open while dimOS drives the arm.
3. Keep the emergency stop within reach.

## Run

```bash
dimos run fr3-planner-coordinator --frankaconnection.ip 172.16.0.2
```

The planner plans with the FR3 model from Franka's `franka_description`
(pinned in `config.py`) and the Hand's TCP as its tool frame. Its home pose
is Franka's ready pose.

## How it drives the arm

- **Arm.** Joint targets go to libfranka's `AsyncPositionControlHandler`: the
  robot plans toward each target itself, so targets arrive at the
  coordinator's 100 Hz instead of libfranka's 1 kHz callback. Franka marks
  this API experimental. Because the robot plans, the connection does not
  require a real-time kernel (`enforce_realtime=False`, as in Franka's own
  example for this API).
- **Sessions.** Starting an async session makes libfranka send all-zero joint
  positions, which the robot takes as a target, so the connection starts no
  session until the first joint target arrives, then immediately sends the
  measured pose as the target. A session ends by cancelling it (a StopMove),
  never with `stop_control()`, which finishes with all-zero positions too.
  While a session runs, a drift guard stops the robot if a joint leaves the
  span between where it was and its latest target by more than 0.02 rad.
- **Sessions end at their target.** An async session is a point-to-point
  motion: once the arm reaches its target and settles, the robot ends it, and
  libfranka reports "Unexpected reply to a Move command" through the session's
  feedback (not through `Robot.read_once()`), so the connection checks that
  feedback every cycle. Ended at the target, it is a finished session and the
  next new target starts another; a target within 0.01 rad of the arm starts
  none. Ended anywhere else, or with the arm standing still over 0.02 rad from
  its target for 0.3 s, the connection faults.
- **Speed.** The robot's planner is capped at `speed_fraction` of each joint's
  top speed (default 0.25, the value Franka's example uses). The model's
  velocity limits are scaled by the same share, so plans and the arm keep
  pace. Change the share in both places together (`speed_fraction` on the
  connection and in `fr3_model` / `make_fr3_model_config`), or a slower arm
  falls behind its plan and the jump guard halts it.
- **Safety checks.** A target more than `jump_limit_s` (0.1 s) of travel at
  the scaled speed from the measured position is refused, as is any target
  while the latest reading is older than 0.1 s. If commands stop for
  `deadman_timeout_s`, the arm is held where it is. Collision thresholds are
  set to Franka's example values (20/40 Nm, 10/20 N).
- **Errors.** A reflex (collision, limit) or a stop from outside dimOS ends the
  control session and faults the connection, with libfranka's error names in
  `status().last_error`. Call `clear_errors()` (libfranka's automatic error
  recovery, as Desk's recovery does) once it is safe.
- **Hand.** The gripper joint is `arm/fr3_finger_joint1`, 0.0 (closed) to
  0.04 m (open) per finger. Closing calls `grasp()` with `grasp_force`
  (default 20 N), which holds an object rather than failing when the fingers
  are blocked; opening calls `move()`.

## Not supported yet

- Velocity, Cartesian and torque commands (positions only).
- Simulation: there is no FR3 MuJoCo scene, so `--simulation` is not wired.
