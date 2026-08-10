# Long-Horizon Sorting Cell

**FFAI Robothon Summer 2026 submission**

| | |
|---|---|
| **Project name** | Long-Horizon Sorting Cell |
| **Registration UUID** | `d127863f-405f-444a-bb1f-13a610def6ba` |
| **Participant** | Iihyikk |
| **AI tools used** | Claude |
| **Direction** | Long-horizon tasks |
| **Robot platform** | Custom 6-DoF arm with a coupled parallel-jaw gripper, authored in MJCF (primitive geoms only, no external meshes) |

---

## Task goal

Six colour-coded cubes start scattered on a workbench. The robot has to clear
the bench by placing every cube into the bin that matches its colour. A full
episode chains roughly fifty sub-goals - select, approach, descend, grasp,
lift, transfer, align, release, retreat, repeat - and the arm has to recover on
its own from grasps that miss and from cubes that slip mid-transfer. That
closed-loop chaining, rather than a single scripted pick, is what makes the
task long-horizon.

## Technical approach

The stack is deliberately thin and readable, four layers with one job each.

**Model.** `scene/arm.xml` describes the manipulator: six hinge joints, a
gripper base and two prismatic jaws coupled by an `equality/joint` constraint
so one actuator drives both fingers. `scene/sorting_cell.xml` includes the arm
and builds the world around it - workbench, three bins, six free cubes, three
fixed cameras and a wrist camera. The simulation runs at a 2 ms timestep with
the `implicitfast` integrator and an elliptic friction cone, which keeps the
stiff grasp contacts well behaved.

**Kinematics.** A damped least-squares differential IK solver turns a Cartesian
goal into a joint target. It stacks the positional and rotational site
Jacobians from `mj_jacSite`, adds a null-space term that biases the arm back
towards its home posture, and clips to the joint limits at every iteration. The
solver runs on a scratch `MjData`, so planning never disturbs the physics.

**Control.** The IK output is not written to `data.ctrl` directly. A
rate-limited joint tracker interpolates towards it at a bounded joint speed,
turning discrete goals into a smooth trajectory. The gripper controller ramps
the jaw actuator and reports whether an object is genuinely held, using the two
jaw touch sensors together with the residual jaw travel.

**Task.** An eleven-state machine sequences the episode. Grasps are verified
before the arm lifts, the hold is re-checked during lift and transfer, and every
state carries a watchdog timeout. Any failure routes to a `RECOVER` state; a
cube is abandoned after three attempts so a single awkward object can never
deadlock the run.

See `docs/ARCHITECTURE.md` for a file-by-file walkthrough.

## Core features

- Fully self-contained MJCF workcell - no meshes, no downloads, compiles in
  milliseconds.
- Damped least-squares IK with null-space posture control and joint-limit
  clipping.
- Coupled parallel-jaw gripper driven by one actuator via an equality
  constraint.
- Sensor-based grasp verification and slip detection (jaw touch sensors, wrist
  force/torque, joint states).
- Long-horizon state machine with retries, per-state watchdogs and graceful
  abandonment.
- Seeded layout randomisation, so an episode is reproducible from `--seed`.
- Built-in data collection: every run writes a compressed trajectory dataset
  (observations, actions, cube poses, wrist images) plus a JSON report.
- Video recording streamed straight to MP4, so long episodes do not exhaust RAM.
- Smoke tests covering model compilation, IK round-trip and state-machine
  progress.

## Highlights

The part worth looking at is the failure handling. Most scripted MuJoCo pick
and place demos assume the grasp succeeded; here the touch sensors decide, and
the machine is written so that *every* transition out of a manipulation state
is guarded. Combined with the per-state watchdogs, that is what lets six cubes
be sorted in one uninterrupted episode under a randomised layout.

The second is that the whole cell is authored from primitives. Anyone can clone
the fork, run one command and get an identical scene - there is nothing to
download and nothing to convert.

## Quick start

```bash
git clone https://github.com/Iihyikk/Robothon-starter.git
cd Robothon-starter/submissions/longhorizon-sorting-cell
python3 -m pip install -r requirements.txt

# 1. Verify the model compiles (no renderer needed)
python run_demo.py --check-scene

# 2. Headless run: writes demo.mp4, trajectory.npz and report.json
python run_demo.py

# 3. Watch it live in the MuJoCo viewer
python run_demo.py --viewer --no-video

# 4. Three randomised episodes, useful as a small benchmark
python run_demo.py --episodes 3 --seed 7
```

Artifacts land in `outputs/<timestamp>-episodeNN/`:

| File | Contents |
|---|---|
| `demo.mp4` | Rendered episode from the `overview` camera |
| `trajectory.npz` | Time, state, joint states, actions, EE pose, touch forces, cube poses, wrist RGB |
| `report.json` | Per-cube outcome, attempt counts, timings, dataset schema |

### Useful flags

| Flag | Meaning |
|---|---|
| `--camera front` / `--camera topdown` | Render from another fixed camera |
| `--fps 60 --width 1920 --height 1080` | Higher quality video |
| `--no-randomize` | Use the deterministic cube layout from the MJCF |
| `--max-seconds 180` | Shorten the episode time budget |
| `--no-video` | Dataset only, no rendering |

### Tests

```bash
python3 -m pip install pytest
python -m pytest tests -q
```

## Demo video

The demo video is produced by the submitted code itself. Running
`python run_demo.py` renders the whole episode - startup, the workbench and
bins, each grasp and transfer, and the final sorted state - to
`outputs/<timestamp>-episode01/demo.mp4` at 1280x720 and 30 fps. A single
episode is comfortably inside the recommended one to three minute window. The
MP4 is not committed to the repository because generated binaries do not belong
in the fork; regenerate it with the one command above.

## Repository layout

```
submissions/longhorizon-sorting-cell/
  README.md              this file
  registration.json      participant UUID and project metadata
  requirements.txt       mujoco, numpy, imageio
  run_demo.py            entry point: check-scene / headless / viewer modes
  docs/ARCHITECTURE.md   file-by-file design notes
  scene/arm.xml          6-DoF arm and coupled parallel-jaw gripper
  scene/sorting_cell.xml workbench, bins, cubes, cameras, sensors
  sortbot/config.py      typed configuration for every stage
  sortbot/kinematics.py  damped least-squares differential IK
  sortbot/controller.py  rate-limited arm and gripper controllers
  sortbot/task.py        long-horizon sorting state machine
  sortbot/recorder.py    video and trajectory dataset recording
  tests/                 smoke tests
```

## Current limitations

- Perception is privileged: cube poses are read from the simulator rather than
  estimated from the cameras, so the wrist images are recorded but not yet used
  in the control loop.
- Grasps are always top-down. Cubes that end up wedged against a bin wall are
  abandoned after three attempts instead of being re-oriented.
- The controller is position-based. There is no compliance or force control, so
  the gripper squeezes to a fixed travel rather than to a target force.
- Contact and actuator gains were tuned by hand for this cube size; very light
  or very heavy objects would need retuning.

## Future improvements

- Close the perception loop by estimating cube poses from the wrist and topdown
  cameras instead of reading them from the simulator.
- Add a small planner that chooses the grasp axis from the cube pose, so cubes
  near a wall can be approached from the side.
- Replace the fixed-travel squeeze with a force-tracking grasp using the wrist
  force/torque sensor.
- Use the recorded `trajectory.npz` files to train a behaviour-cloning policy
  and compare it against the scripted state machine on the same seeds.
- Extend the task with stacking and with objects of mixed shape, which would
  stress the long-horizon recovery logic much harder.
