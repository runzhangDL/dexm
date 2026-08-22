import newton
import numpy as np
import warp as wp
from pathlib import Path


FRANKA_FINGER_CLOSED = 0.0
FRANKA_FINGER_OPEN = 0.04

FRANKA_DOF_COUNT = 9
FRANKA_ARM_DOF_COUNT = 7

# These values set controller limits and solver regularization for the first 9 Franka DOFs.
FRANKA_EFFORT_LIMITS = [87, 87, 87, 87, 12, 12, 12, 100, 100]
FRANKA_MUJOCO_ARMATURE = [0.195] * 4 + [0.074] * 3 + [0.1] * 2

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
FRANKA_URDF_PATH = ASSETS_DIR / "franka_emika_panda" / "urdf" / "fr3_franka_hand.urdf"

FRANKA_JOINT_LOWER = [-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159, 0.0, 0.0]
FRANKA_JOINT_UPPER = [2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159, 0.04, 0.04]

FRANKA_TARGET_KE = [900, 900, 700, 700, 400, 400, 400, 100, 100]
FRANKA_TARGET_KD = [90, 90, 70, 70, 40, 40, 40, 10, 10]

FRANKA_VELOCITY_LIMITS = [2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610, 0.1, 0.1]

def sample_random_franka_q(rng: np.random.Generator) -> np.ndarray:
    lower = np.asarray(FRANKA_JOINT_LOWER, dtype=np.float32)
    upper = np.asarray(FRANKA_JOINT_UPPER, dtype=np.float32)
    return rng.uniform(lower, upper).astype(np.float32)

def sample_random_franka_qd(rng: np.random.Generator, scale: float = 0.5) -> np.ndarray:
    vel_limits = np.asarray(FRANKA_VELOCITY_LIMITS, dtype=np.float32) * scale
    return rng.uniform(-vel_limits, vel_limits).astype(np.float32)

def random_arm_pos_scene(rng: np.random.Generator):
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.gap = 0.0
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

    table_height = 0.1
    table_pos = wp.vec3(0.0, -0.5, 0.5 * table_height)
    table_top_center = table_pos + wp.vec3(0.0, 0.0, 0.5 * table_height)
    builder.add_shape_box(
        body=-1,
        hx=0.4,
        hy=0.4,
        hz=0.5 * table_height,
        xform=wp.transform(table_pos),
    )

    robot_base_pos = table_top_center + wp.vec3(-0.5, 0.0, 0.0)
    builder.add_urdf(
        str(FRANKA_URDF_PATH),
        xform=wp.transform(robot_base_pos, wp.quat_identity()),
        floating=False,
        enable_self_collisions=False,
        parse_visuals_as_colliders=False,
    )

    builder.joint_q[:FRANKA_DOF_COUNT] = sample_random_franka_q(rng=rng)
    builder.joint_effort_limit[:FRANKA_DOF_COUNT] = FRANKA_EFFORT_LIMITS
    builder.joint_armature[:FRANKA_DOF_COUNT] = FRANKA_MUJOCO_ARMATURE

    builder.joint_target_ke[:FRANKA_DOF_COUNT] = FRANKA_TARGET_KE
    builder.joint_target_kd[:FRANKA_DOF_COUNT] = FRANKA_TARGET_KD

    builder.joint_qd[:FRANKA_DOF_COUNT] = sample_random_franka_qd(rng=rng)

    return builder

def generate_worlds(num_worlds:int=2, seed: int = 42):
    scene = newton.ModelBuilder()
    rng = np.random.default_rng(seed)
    for _ in range(num_worlds):
        scene.add_world(random_arm_pos_scene(rng=rng))
    multi_arm_model = scene.finalize()
    return multi_arm_model

def generate_data(num_worlds:int=2, num_shards:int =4, episode_per_shard:int=2, substeps: int = 300, dt: float = 0.02, out_dir:str="dataset/franka_emulator"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for shard_idx in range(num_shards):

        seed = 42 + shard_idx * 100
        rng = np.random.default_rng(seed)

        # 1. Instantiate parallel simulation on GPU
        model = generate_worlds(num_worlds=num_worlds, seed=seed)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()

        collision_pipeline = newton.sim.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        solver = newton.solvers.SolverMuJoCo(model=model)



        buf_q = np.zeros((num_worlds, substeps + 1, FRANKA_DOF_COUNT), dtype=np.float32)
        buf_qd = np.zeros((num_worlds, substeps + 1, FRANKA_DOF_COUNT), dtype=np.float32)
        buf_target = np.zeros((num_worlds, substeps, FRANKA_DOF_COUNT), dtype=np.float32)

        buf_q[:, 0, :] = state_0.joint_q.numpy().reshape(num_worlds, FRANKA_DOF_COUNT)
        buf_qd[:, 0, :] = state_0.joint_qd.numpy().reshape(num_worlds, FRANKA_DOF_COUNT)

        for step in range(substeps):
            # targets = action_gen.step()
            control.joint_target_q.assign(targets.reshape(-1))
            buf_target[:, step, :] = targets

            state_0.clear_forces()
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_in=state_0, state_out=state_1, control=control, contacts=contacts, dt=dt)

            # Record resulting state (t + 1)
            buf_q[:, step + 1, :] = state_1.joint_q.numpy().reshape(num_worlds, FRANKA_DOF_COUNT)
            buf_qd[:, step + 1, :] = state_1.joint_qd.numpy().reshape(num_worlds, FRANKA_DOF_COUNT)

            state_0, state_1 = state_1, state_0


if __name__ == "__main__":
    generate_data()