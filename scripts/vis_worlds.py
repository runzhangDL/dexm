import time
from pathlib import Path
import numpy as np
import warp as wp
import newton

from generate_emulator_data import generate_worlds as generate_multi_arm_model

AXIS_COLORS = wp.array([(1.0, 0.1, 0.1), (0.1, 1.0, 0.1), (0.1, 0.4, 1.0)], dtype=wp.vec3)
AXIS_HALF = 0.05
_AXIS_BASIS = (wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0), wp.vec3(0.0, 0.0, 1.0))


def log_frame_axes(viewer, name: str, pos, quat=None, *, half=AXIS_HALF, width=0.01):
    arr = np.asarray(pos, dtype=np.float32).reshape(-1)
    if arr.size == 7:
        position, rotation = arr[:3], arr[3:7]
    elif arr.size == 3:
        position = arr
        rotation = quat if quat is not None else (0.0, 0.0, 0.0, 1.0)
    else:
        raise ValueError(f"log_frame_axes: pos must be length 3 or 7, got {arr.size}")

    q = wp.quat(*(float(c) for c in rotation))
    starts = np.tile(position, (3, 1))
    ends = np.stack(
        [position + half * np.asarray(wp.quat_rotate(q, b), dtype=np.float32) for b in _AXIS_BASIS]
    )
    viewer.log_lines(name, starts, ends, colors=AXIS_COLORS, width=width)


def visualize_model(model: newton.Model, dt: float = 1.0 / 60.0):
    # 1. Allocate state, control, and contact buffers
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    contacts = model.contacts()

    # 2. SolverMuJoCo instance
    solver = newton.solvers.SolverMuJoCo(model)

    # 3. Setup Viser viewer
    viewer = newton.viewer.ViewerViser(verbose=True)
    viewer.set_model(model)
    viewer.set_camera(wp.vec3(0.5, 0.0, 0.5), -15, -140)

    print("\n[Viewer Running] Open the URL printed by Viser in your browser.")
    print("Press Ctrl+C in terminal to stop.\n")

    sim_time = 0.0

    try:
        while True:
            t_start = time.perf_counter()

            # 4. Evaluate collisions and step physics with contacts passed
            model.collide(state_in, contacts)
            solver.step(
                state_in=state_in,
                state_out=state_out,
                control=control,
                contacts=contacts,
                dt=dt,
            )

            # 5. Evaluate FK to update body transforms for visualization
            newton.eval_fk(model, state_out.joint_q, state_out.joint_qd, state_out)

            # 6. Render frame to viewer
            viewer.begin_frame(sim_time)
            viewer.log_state(state_out)
            log_frame_axes(viewer, "/debug/world_origin", np.array([0.0, 0.0, 0.0], dtype=np.float32))
            viewer.end_frame()

            # Swap states for next step
            state_in, state_out = state_out, state_in
            sim_time += dt

            # Sync to real-time
            elapsed = time.perf_counter() - t_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping visualizer.")
    finally:
        viewer.close()


if __name__ == "__main__":
    wp.init()
    model = generate_multi_arm_model(num_worlds=2, seed=42)
    visualize_model(model)