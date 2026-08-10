# Architecture

This document explains how the Long-Horizon Sorting Cell is put together and
why each piece exists. It is meant to be read alongside the source; every
section maps onto exactly one file.

## 1. Scene (`scene/`)

The workcell is described by two MJCF files so that the robot can be reused in
other scenes.

**`scene/arm.xml`** defines a 6-DoF serial arm and a coupled parallel-jaw
gripper. Everything is built from primitive geoms (capsules, boxes, a
cylinder), so the model has no external mesh dependencies and compiles in a few
milliseconds. Notable modelling choices:

- Default classes (`arm`, `jaw`) keep joint damping, armature, contact
  parameters and actuator gains in one place.
- The two jaws each have their own slide joint; an `equality/joint` constraint
  with `polycoef="0 1 0 0 0"` mirrors the right jaw onto the left one, so a
  single position actuator drives the whole gripper.
- The pads use `condim="4"`, high friction and `priority="2"` so that grasp
  contacts dominate over the cube-table contacts.
- Sensors expose everything the controller needs: end-effector pose and linear
  velocity, a wrist force/torque pair, two jaw touch sensors, and joint
  position/velocity for all seven actuated degrees of freedom.

**`scene/sorting_cell.xml`** includes the arm and adds the world: a textured
floor, a workbench, three colour-coded bins built from five box geoms each, six
free-floating cubes, three fixed cameras (`overview`, `front`, `topdown`) and
per-cube `framepos` sensors. The integrator is `implicitfast` with an elliptic
friction cone, which keeps stiff grasp contacts stable at a 2 ms timestep.

## 2. Kinematics (`sortbot/kinematics.py`)

`DifferentialIK` converts a Cartesian goal into a joint target using damped
least squares:

    dq = (J^T J + lambda^2 I)^-1 (J^T e + k N (q_rest - q))

where `J` stacks the positional and rotational site Jacobians from
`mj_jacSite`, `e` is the 6-vector pose error, `N` is the null-space projector
and `q_rest` is the home posture. Two details matter in practice:

1. The solver iterates on a scratch `MjData`, so it never perturbs the live
   simulation - the physics state and the planner stay decoupled.
2. Joint limits are enforced by clipping after each update and the per-iteration
   step is capped, which keeps the arm out of the wrist singularity when it
   reaches across the bench.

`top_down_quat(yaw)` builds the grasp orientation: the gripper approach axis is
flipped to point straight down and then spun about the vertical to line the jaws
up with the cube.

## 3. Control (`sortbot/controller.py`)

`ArmController` never writes an IK solution straight into `data.ctrl`. It
keeps an internal command vector and moves it towards the target at a bounded
joint speed, which turns the discrete IK output into a smooth trajectory and
stops the position actuators from being step-excited.

`GripperController` ramps the single gripper actuator between an open and a
closed travel and answers the only question the task really cares about:
*is something actually being held?* That is decided from the two jaw touch
sensors plus the residual jaw travel, so closing on empty air is detected
immediately instead of being discovered after a failed transfer.

## 4. Task (`sortbot/task.py`)

The sorting task is a finite state machine over eleven states. A normal cube
follows `SELECT -> APPROACH -> DESCEND -> GRASP -> LIFT -> TRANSFER -> ALIGN ->
RELEASE -> RETREAT`, and six cubes chain about fifty sub-goals into one
episode. Robustness comes from three mechanisms:

- **Grasp verification.** `GRASP` only advances if the touch sensors confirm a
  hold; otherwise the cube attempt counter is incremented and the machine goes
  to `RECOVER`.
- **Slip detection.** `LIFT` and `TRANSFER` keep checking the hold, so a cube
  that slips is re-planned instead of being released into empty space.
- **Per-state watchdogs.** Every state has a timeout; exceeding it also routes
  to `RECOVER`. After three failures a cube is abandoned and the episode
  continues, so one hard cube can never deadlock the run.

`evaluate()` closes the loop with a purely geometric check - where did each
cube actually end up - rather than trusting the state machine bookkeeping.

## 5. Recording (`sortbot/recorder.py`)

`EpisodeRecorder` streams video frames straight to an MP4 writer so a long
episode never has to be buffered in RAM, and simultaneously builds a trajectory
table: time, state id, arm positions and velocities, commanded actions, gripper
travel, end-effector pose, jaw touch forces and all six cube positions. The
table is written as a compressed `trajectory.npz` next to a `report.json`
summary, which makes the run directory a small self-contained dataset.

## 6. Reproducibility

The layout randomiser is driven by a `numpy.random.Generator` seeded from
`--seed`, so a given seed always produces the same episode. `--check-scene`
compiles the model and prints its statistics without needing a renderer, which
is the quickest way for a reviewer to confirm the model is valid.
