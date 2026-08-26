import newton
from newton.solvers import SolverMuJoCo, SolverVBD
import warp as wp
from pathlib import Path
import numpy as np
import math

# noinspection unresolved-references
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledProxy

ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"
FRANKA_URDF_PATH = str(
    ASSETS_DIR / "franka_emika_panda" / "urdf" / "fr3_franka_hand.urdf"
)

FRANKA_Q = [
    -3.6802115e-03,
    2.3901723e-02,
    3.6804110e-03,
    -2.3683236e00,
    -1.2918962e-04,
    2.3922248e00,
    7.8549200e-01,
    0.04,
    0.04,
]

CABLE_CENTER = wp.vec3(0.0, 0.0, 0.003)
CABLE_LENGTH = 0.85

CABLE_RADIUS = 0.003


class DexmBuilder:
    def __init__(
        self,
        worlds_count: int = 1,
        fps: int = 30,
        sim_substeps: int = 20,
        use_graph: bool = False,
    ):
        self.worlds_count = worlds_count
        self.fps = fps
        self.sim_substeps = sim_substeps

        self._build_scene()

        self.use_graph = use_graph
        self.control = self.model.control()
        self._build_solvers()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="explicit",
            shape_pairs_filtered=self._ground_shape_pairs(),
        )
        self.contacts = self.collision_pipeline.contacts()
        self.solver.prepare_contacts(self.contacts)

        newton.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, self.state_0
        )
        newton.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, self.state_1
        )

    def _ground_shape_pairs(self) -> wp.array:
        dynamic_shapes = (
            set(self.franka1_shapes)
            | set(self.cable_shapes)
            | set(self.franka2_shapes)
            | set(self.cube_shapes)
        )
        ground_shapes = set(self.ground_shapes)
        arm_shapes = set(self.franka1_shapes) | set(self.franka2_shapes)
        cube_shapes = set(self.cube_shapes)
        pairs = [
            (int(a), int(b))
            for a, b in self.model.shape_contact_pairs.numpy()
            if (
                ({int(a), int(b)} & dynamic_shapes)
                and ({int(a), int(b)} & ground_shapes)
            )
            or (({int(a), int(b)} & arm_shapes) and ({int(a), int(b)} & cube_shapes))
            or (
                ({int(a), int(b)} & set(self.franka1_shapes))
                and ({int(a), int(b)} & set(self.franka2_shapes))
            )
        ]
        if not pairs:
            raise RuntimeError("No robot- or cable-ground contact pairs were generated")
        return wp.array(
            np.asarray(pairs, dtype=np.int32), dtype=wp.vec2i, device=self.model.device
        )

    def _build_solvers(self):
        self.solver = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupled.Entry(
                    name="mjc",
                    solver=lambda v: SolverMuJoCo(
                        model=v,
                        solver="newton",
                        integrator="implicitfast",
                        cone="elliptic",
                        iterations=100,
                        ls_iterations=20,
                        use_mujoco_contacts=False,
                        njmax=max(256, 128 * self.worlds_count),
                        nconmax=max(256, 128 * self.worlds_count),
                    ),
                    bodies=self.franka1_bodies + self.franka2_bodies + self.cube_bodies,
                    joints=self.franka1_joints + self.franka2_joints + self.cube_joints,
                ),
                SolverCoupled.Entry(
                    name="vbd",
                    solver=lambda v: SolverVBD(
                        model=v,
                        iterations=int(1.5 * self.cable_num_segments),
                        rigid_avbd_beta=1.0e2,
                        rigid_contact_k_start=1.0e3,
                        rigid_contact_history=True,
                    ),
                    bodies=self.cable_bodies,
                    joints=self.cable_joints,
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="mjc",
                        destination="vbd",
                        bodies=self.franka1_bodies
                        + self.franka2_bodies
                        + self.cube_bodies,
                        mass_scale=1.0,
                        mode="lagged",
                        collision_pipeline=lambda model: newton.CollisionPipeline(
                            model,
                            broad_phase="explicit",
                            contact_matching="latest",
                            deterministic=True,  # item 18, free here
                            rigid_contact_max=30000,  # item 2 — 70 segments is ~2346 cable-cable pairs
                        ),
                        collide_interval=1,
                    ),
                ],
                iterations=int(4),
            ),
        )

    def _build_scene(self):
        template = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        template.rigid_gap = 0.01
        SolverMuJoCo.register_custom_attributes(template)
        SolverVBD.register_custom_attributes(template)
        self._add_one_arm(template, 1)
        self._add_one_arm(template, 2)

        gravcomp = template.custom_attributes["mujoco:gravcomp"]
        if gravcomp.values is None:
            gravcomp.values = {}
        for body in self.franka1_bodies + self.franka2_bodies:
            gravcomp.values[body] = 1.0

        self._add_cable(template)

        self._add_cube(template)

        bodies_per_world = template.body_count
        joints_per_world = template.joint_count
        shapes_per_world = template.shape_count

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.rigid_gap = template.rigid_gap
        builder.replicate(template, world_count=self.worlds_count)
        self._expand_world_indices(bodies_per_world, joints_per_world, shapes_per_world)

        plane_cfg = newton.ModelBuilder.ShapeConfig(
            ke=1.0e4, kd=100.0, mu=0.6, margin=0.0, gap=0.01
        )
        # surface_z = float(CABLE_CENTER[2]) - 0.005  # radius = 0.005
        surface_z = 0.0
        self.ground_shapes = [
            builder.add_ground_plane(
                height=surface_z, cfg=plane_cfg, label="ground_plane"
            )
        ]
        builder.color()

        self.model = builder.finalize()

        SOLREF_MODE_RAW = 1  # newton._src.solvers.mujoco.constants
        sl = self.model.mujoco.solreflimit.numpy()
        sm = self.model.mujoco.solreflimit_mode.numpy()
        sl[:] = (0.02, 1.0)
        sm[:] = SOLREF_MODE_RAW
        self.model.mujoco.solreflimit.assign(sl)
        self.model.mujoco.solreflimit_mode.assign(sm)

        self.device = self.model.device

    def _add_cube(self, template):
        cube_body_start = template.body_count
        cube_joint_start = template.joint_count
        cube_shape_start = template.shape_count

        CUBE_SIZE = 0.06
        CUBE_DENSITY = 2300.0
        cube_pos = wp.vec3(0.0, 0.1, 0.5 * CUBE_SIZE)
        cube_body = template.add_link(xform=wp.transform(cube_pos, wp.quat_identity()))
        template.add_shape_box(
            body=cube_body,
            hx=0.5 * CUBE_SIZE,
            hy=0.5 * CUBE_SIZE,
            hz=0.5 * CUBE_SIZE,
            cfg=newton.ModelBuilder.ShapeConfig(
                mu=0.5, density=CUBE_DENSITY, ke=1.0e4, kd=1.0e2
            ),
        )
        cube_joint = template.add_joint_free(child=cube_body)
        template.add_articulation([int(cube_joint)], label="cube")

        self.cube_bodies = list(range(cube_body_start, template.body_count))
        self.cube_joints = list(range(cube_joint_start, template.joint_count))
        self.cube_shapes = list(range(cube_shape_start, template.shape_count))

    def _expand_world_indices(
        self, bodies_per_world, joints_per_world, shapes_per_world
    ):
        def expand(ids: list[int], stride: int) -> list[int]:
            return [
                world * stride + id_
                for world in range(self.worlds_count)
                for id_ in ids
            ]

        self.franka1_bodies = expand(self.franka1_bodies, bodies_per_world)
        self.franka1_joints = expand(self.franka1_joints, joints_per_world)
        self.franka1_shapes = expand(self.franka1_shapes, shapes_per_world)
        self.franka2_bodies = expand(self.franka2_bodies, bodies_per_world)
        self.franka2_joints = expand(self.franka2_joints, joints_per_world)
        self.franka2_shapes = expand(self.franka2_shapes, shapes_per_world)
        self.gripper1_bodies = expand(self.gripper1_bodies, bodies_per_world)
        self.gripper2_bodies = expand(self.gripper2_bodies, bodies_per_world)
        self.cable_bodies = expand(self.cable_bodies, bodies_per_world)
        self.cable_joints = expand(self.cable_joints, joints_per_world)
        self.cable_shapes = expand(self.cable_shapes, shapes_per_world)
        self.cube_bodies = expand(self.cube_bodies, bodies_per_world)
        self.cube_joints = expand(self.cube_joints, joints_per_world)
        self.cube_shapes = expand(self.cube_shapes, shapes_per_world)

    def _add_cable(self, template):
        cable_body_start = template.body_count
        cable_joint_start = template.joint_count
        cable_shape_start = template.shape_count

        cable_cfg = newton.ModelBuilder.ShapeConfig(
            density=1000.0, ke=1.0e4, kd=150, mu=0.45, margin=0.0, gap=0.005
        )

        self.cable_num_segments = int(CABLE_LENGTH / 0.012)
        points, quats = newton.utils.create_straight_cable_points_and_quaternions(
            start=CABLE_CENTER - wp.vec3(0.5 * CABLE_LENGTH, 0.0, 0.0),
            direction=wp.vec3(1.0, 0.0, 0.0),
            length=CABLE_LENGTH,
            num_segments=self.cable_num_segments,
            twist_total=0.0,
        )

        CABLE_E = 3.0e5  # Pa
        seg_len = CABLE_LENGTH / self.cable_num_segments
        area_moment = math.pi * CABLE_RADIUS**4 / 4.0
        bend_stiffness = CABLE_E * area_moment / seg_len

        stretch_stiffness = 1.0e6
        stretch_damping = 1.0

        # bend_stiffness = 5.0e-5
        bend_damping = 2.0e-2 * bend_stiffness

        twist_stiffness = (2.0 / 3.0) * bend_stiffness
        twist_damping = 2.0e-2 * twist_stiffness

        template.add_rod(
            positions=points,
            quaternions=quats,
            radius=CABLE_RADIUS,
            body_frame_origin="start",
            cfg=cable_cfg,
            stretch_stiffness=stretch_stiffness,
            stretch_damping=stretch_damping,
            bend_stiffness=bend_stiffness,
            bend_damping=bend_damping,
            twist_stiffness=twist_stiffness,
            twist_damping=twist_damping,
            label="vbd_cable",
        )

        self.cable_bodies = list(range(cable_body_start, template.body_count))
        self.cable_joints = list(range(cable_joint_start, template.joint_count))
        self.cable_shapes = list(range(cable_shape_start, template.shape_count))

        self.cable_bodies_count = len(self.cable_bodies)
        self.cable_mid_body_offset = self.cable_bodies_count // 2

    def _add_one_arm(self, template, arm_id: int):
        franka_body_start = template.body_count
        franka_joint_start = template.joint_count
        franka_shape_start = template.shape_count
        franka_dof_start = template.joint_dof_count

        if arm_id == 1:
            pos = wp.vec3(0.65, 0.0, 0.0)
            rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi)
        elif arm_id == 2:
            pos = wp.vec3(-0.65, 0.0, 0.0)
            rot = wp.quat_identity()
        else:
            raise Exception("Unknown arm id, Only support two arms with 1 and 2 as ID")

        template.add_urdf(
            str(FRANKA_URDF_PATH),
            xform=wp.transform(pos, rot),
            floating=False,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
        )

        template.joint_q[franka_dof_start : franka_dof_start + len(FRANKA_Q)] = FRANKA_Q
        template.joint_target_q[franka_dof_start : franka_dof_start + len(FRANKA_Q)] = (
            FRANKA_Q
        )
        template.joint_target_ke[franka_dof_start : franka_dof_start + 7] = [400.0] * 7
        template.joint_target_kd[franka_dof_start : franka_dof_start + 7] = [80.0] * 7
        template.joint_target_ke[franka_dof_start + 7 : franka_dof_start + 9] = [
            1000.0,
            1000.0,
        ]
        template.joint_target_kd[franka_dof_start + 7 : franka_dof_start + 9] = [
            100.0,
            100.0,
        ]
        template.joint_effort_limit[franka_dof_start : franka_dof_start + 4] = [
            87.0
        ] * 4
        template.joint_effort_limit[franka_dof_start + 4 : franka_dof_start + 7] = [
            12.0
        ] * 3
        template.joint_effort_limit[franka_dof_start + 7 : franka_dof_start + 9] = [
            140.0,
            140.0,
        ]
        template.joint_armature[franka_dof_start : franka_dof_start + 7] = [1.0e-3] * 7
        template.joint_armature[franka_dof_start + 7 : franka_dof_start + 9] = [
            0.0,
            0.0,
        ]
        GRIP_MIN = 0.9 * CABLE_RADIUS  # ~0.0027, leaves a little pinch preload
        template.joint_limit_lower[franka_dof_start + 7 : franka_dof_start + 9] = [
            GRIP_MIN,
            GRIP_MIN,
        ]

        if arm_id == 1:
            self.franka1_bodies = list(range(franka_body_start, template.body_count))
            self.franka1_joints = list(range(franka_joint_start, template.joint_count))
            self.franka1_shapes = list(range(franka_shape_start, template.shape_count))
            self.gripper1_bodies = [
                body
                for body in self.franka1_bodies
                if "hand" in template.body_label[body]
                or "finger" in template.body_label[body]
            ]
            if not self.gripper1_bodies:
                raise RuntimeError("No gripper1 bodies found")
        elif arm_id == 2:
            self.franka2_bodies = list(range(franka_body_start, template.body_count))
            self.franka2_joints = list(range(franka_joint_start, template.joint_count))
            self.franka2_shapes = list(range(franka_shape_start, template.shape_count))
            self.gripper2_bodies = [
                body
                for body in self.franka2_bodies
                if "hand" in template.body_label[body]
                or "finger" in template.body_label[body]
            ]
            if not self.gripper2_bodies:
                raise RuntimeError("No gripper2 bodies found")


if __name__ == "__main__":
    print("Hello World")
    dexm = DexmBuilder()
