import newton
from newton.solvers import SolverMuJoCo, SolverVBD
from newton.sensors import SensorTiledCamera
import warp as wp
from pathlib import Path
import numpy as np
import math

# noinspection unresolved-references
from newton.solvers.experimental.coupled import (
    SolverCoupled,
    SolverCoupledProxy,
)

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
        sim_substeps: int = 10,
        use_graph: bool = False,
        arm_separation: float = 0.80,
        cable_center: wp.vec3 | tuple | None = None,
        cable_length: float = 0.85,
        cable_radius: float = 0.004,
        cube_pos: wp.vec3 | tuple | None = None,
        cube_size: float = 0.06,
        enable_camera: bool = True,
        camera_pos: wp.vec3 | tuple = (0.0, 1.25, 0.65),
        camera_target: wp.vec3 | tuple = (0.0, 0.35, 0.18),
        camera_width: int = 800,
        camera_height: int = 600,
        camera_fov: float = 52.0,
    ):
        self.worlds_count = worlds_count
        self.fps = fps
        self.sim_substeps = sim_substeps
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.arm_separation = arm_separation
        self.enable_camera = enable_camera
        self.camera_pos = np.array(camera_pos, dtype=np.float32)
        self.camera_target = np.array(camera_target, dtype=np.float32)
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fov = camera_fov

        if cable_center is None:
            self.cable_center = wp.vec3(0.0, 0.50, 0.40)
        elif isinstance(cable_center, (tuple, list)):
            self.cable_center = wp.vec3(*cable_center)
        else:
            self.cable_center = cable_center

        self.cable_length = cable_length
        self.cable_radius = cable_radius

        if cube_pos is None:
            self.cube_pos = wp.vec3(0.0, 0.50, 0.20)
        elif isinstance(cube_pos, (tuple, list)):
            self.cube_pos = wp.vec3(*cube_pos)
        else:
            self.cube_pos = cube_pos

        self.cube_size = cube_size

        self._build_scene()

        self.use_graph = use_graph
        self.graph = None
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

        self.init_joint_q = wp.clone(self.model.joint_q)
        self.init_joint_qd = wp.clone(self.model.joint_qd)

        newton.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, self.state_0
        )
        newton.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, self.state_1
        )

        if self.enable_camera:
            self._setup_camera()

        if self.use_graph:
            self.capture()

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
            or (
                ({int(a), int(b)} & arm_shapes)
                and ({int(a), int(b)} & cube_shapes)
            )
            or (
                ({int(a), int(b)} & set(self.franka1_shapes))
                and ({int(a), int(b)} & set(self.franka2_shapes))
            )
        ]
        if not pairs:
            raise RuntimeError(
                "No robot- or cable-ground contact pairs were generated"
            )
        return wp.array(
            np.asarray(pairs, dtype=np.int32),
            dtype=wp.vec2i,
            device=self.model.device,
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
                        iterations=25,
                        ls_iterations=5,
                        use_mujoco_contacts=False,
                        njmax=max(2048, 1024 * self.worlds_count),
                        nconmax=max(1024, 512 * self.worlds_count),
                    ),
                    bodies=self.franka1_bodies
                    + self.franka2_bodies
                    + self.cube_bodies,
                    joints=self.franka1_joints
                    + self.franka2_joints
                    + self.cube_joints,
                ),
                SolverCoupled.Entry(
                    name="vbd",
                    solver=lambda v: SolverVBD(
                        model=v,
                        iterations=30,
                        rigid_avbd_beta=1.0e2,
                        rigid_contact_k_start=1.0e3,
                        rigid_contact_history=True,
                        rigid_body_contact_buffer_size=256,
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
                        collision_pipeline=lambda model: (
                            newton.CollisionPipeline(
                                model,
                                broad_phase="explicit",
                                contact_matching="latest",
                                deterministic=True,
                                rigid_contact_max=10000,
                            )
                        ),
                        collide_interval=1,
                    ),
                ],
                iterations=1,
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
        # self._add_cloth(template)

        self._add_cube(template)

        bodies_per_world = template.body_count
        joints_per_world = template.joint_count
        shapes_per_world = template.shape_count

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.rigid_gap = template.rigid_gap
        builder.replicate(template, world_count=self.worlds_count)
        self._expand_world_indices(
            bodies_per_world, joints_per_world, shapes_per_world
        )

        plane_cfg = newton.ModelBuilder.ShapeConfig(
            ke=1.0e4, kd=100.0, mu=0.6, margin=0.0, gap=0.01
        )
        # surface_z = float(CABLE_CENTER[2]) - 0.005  # radius = 0.005
        surface_z = 0.0
        self.ground_shapes = [
            builder.add_ground_plane(
                height=surface_z,
                cfg=plane_cfg,
                label="ground_plane",
                color=(0.78, 0.81, 0.85),
            )
        ]
        builder.color()
        # Ensure ground plane retains clean neutral studio color
        colors_np = builder.shape_color
        if ground_shape_idx := self.ground_shapes[0]:
            colors_np[ground_shape_idx] = [0.78, 0.81, 0.85]

        self.model = builder.finalize()

        SOLREF_MODE_RAW = 1  # newton._src.solvers.mujoco.constants
        sl = self.model.mujoco.solreflimit.numpy()
        sm = self.model.mujoco.solreflimit_mode.numpy()
        sl[:] = (0.02, 1.0)
        sm[:] = SOLREF_MODE_RAW
        self.model.mujoco.solreflimit.assign(sl)
        self.model.mujoco.solreflimit_mode.assign(sm)

        self.device = self.model.device

    # def _add_cloth(self, template):
    #     cloth_body_start = template.body_count
    #     cloth_joint_start = template.joint_count
    #     cloth_shape_start = template.shape_count
    #
    #     tri_ke = 1.0e5
    #     edge_ke = 0.01
    #     template.add_cloth_grid(
    #         pos=wp.vec3(-0.5, -0.5, 1.0),
    #         rot=wp.quat_identity(),
    #         vel=wp.vec3(0.0),
    #         fix_left=False,
    #         fix_right=False,
    #         dim_x=30,
    #         dim_y=30,
    #         cell_x=1.0 / 30.0,
    #         cell_y=1.0 / 30.0,
    #         mass=0.1,
    #         tri_ke=tri_ke,
    #         tri_ka=tri_ke,
    #         tri_kd=1.0e-2 * tri_ke,
    #         edge_ke=edge_ke,
    #         edge_kd=1.0e-2 * edge_ke,
    #         particle_radius=0.01,
    #     )
    #
    #     self.cloth_bodies = list(range(cloth_body_start, template.body_count))
    #     self.cloth_joints = list(range(cloth_joint_start, template.joint_count))
    #     self.cloth_shapes = list(range(cloth_shape_start, template.shape_count))

    def _add_cube(self, template):
        cube_body_start = template.body_count
        cube_joint_start = template.joint_count
        cube_shape_start = template.shape_count

        cube_body = template.add_link(
            xform=wp.transform(self.cube_pos, wp.quat_identity())
        )
        template.add_shape_box(
            body=cube_body,
            hx=0.5 * self.cube_size,
            hy=0.5 * self.cube_size,
            hz=0.5 * self.cube_size,
            cfg=newton.ModelBuilder.ShapeConfig(
                mu=0.8, density=2300.0, ke=1.0e4, kd=1.0e2
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
            density=1000.0, ke=1.0e4, kd=50.0, mu=0.8, margin=0.0, gap=0.005
        )

        self.cable_num_segments = int(self.cable_length / 0.010)
        points, quats = (
            newton.utils.create_straight_cable_points_and_quaternions(
                start=self.cable_center
                - wp.vec3(0.5 * self.cable_length, 0.0, 0.0),
                direction=wp.vec3(1.0, 0.0, 0.0),
                length=self.cable_length,
                num_segments=self.cable_num_segments,
                twist_total=0.0,
            )
        )

        CABLE_E = 3.0e5  # Pa
        seg_len = self.cable_length / self.cable_num_segments
        area_moment = math.pi * self.cable_radius**4 / 4.0
        bend_stiffness = CABLE_E * area_moment / seg_len

        stretch_stiffness = 1.0e6
        stretch_damping = 1.0

        bend_damping = 2.0e-2 * bend_stiffness

        twist_stiffness = (2.0 / 3.0) * bend_stiffness
        twist_damping = 2.0e-2 * twist_stiffness

        template.add_rod(
            positions=points,
            quaternions=quats,
            radius=self.cable_radius,
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

        half_sep = 0.5 * self.arm_separation
        rot_facing_y = wp.quat_from_axis_angle(
            wp.vec3(0.0, 0.0, 1.0), float(np.pi / 2)
        )
        if arm_id == 1:
            pos = wp.vec3(half_sep, 0.0, 0.0)
            rot = rot_facing_y
        elif arm_id == 2:
            pos = wp.vec3(-half_sep, 0.0, 0.0)
            rot = rot_facing_y
        else:
            raise Exception(
                "Unknown arm id, Only support two arms with 1 and 2 as ID"
            )

        template.add_urdf(
            str(FRANKA_URDF_PATH),
            xform=wp.transform(pos, rot),
            floating=False,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
        )

        template.joint_q[
            franka_dof_start : franka_dof_start + len(FRANKA_Q)
        ] = FRANKA_Q
        template.joint_target_q[
            franka_dof_start : franka_dof_start + len(FRANKA_Q)
        ] = FRANKA_Q
        template.joint_target_ke[franka_dof_start : franka_dof_start + 7] = [
            400.0
        ] * 7
        template.joint_target_kd[franka_dof_start : franka_dof_start + 7] = [
            80.0
        ] * 7
        template.joint_target_ke[
            franka_dof_start + 7 : franka_dof_start + 9
        ] = [
            1000.0,
            1000.0,
        ]
        template.joint_target_kd[
            franka_dof_start + 7 : franka_dof_start + 9
        ] = [
            100.0,
            100.0,
        ]
        template.joint_effort_limit[franka_dof_start : franka_dof_start + 4] = [
            87.0
        ] * 4
        template.joint_effort_limit[
            franka_dof_start + 4 : franka_dof_start + 7
        ] = [12.0] * 3
        template.joint_effort_limit[
            franka_dof_start + 7 : franka_dof_start + 9
        ] = [
            140.0,
            140.0,
        ]
        template.joint_armature[franka_dof_start : franka_dof_start + 7] = [
            1.0e-3
        ] * 7
        template.joint_armature[franka_dof_start + 7 : franka_dof_start + 9] = [
            0.0,
            0.0,
        ]
        GRIP_MIN = 0.9 * CABLE_RADIUS  # ~0.0027, leaves a little pinch preload
        template.joint_limit_lower[
            franka_dof_start + 7 : franka_dof_start + 9
        ] = [
            GRIP_MIN,
            GRIP_MIN,
        ]

        if arm_id == 1:
            self.franka1_bodies = list(
                range(franka_body_start, template.body_count)
            )
            self.franka1_joints = list(
                range(franka_joint_start, template.joint_count)
            )
            self.franka1_shapes = list(
                range(franka_shape_start, template.shape_count)
            )
            self.gripper1_bodies = [
                body
                for body in self.franka1_bodies
                if "hand" in template.body_label[body]
                or "finger" in template.body_label[body]
            ]
            if not self.gripper1_bodies:
                raise RuntimeError("No gripper1 bodies found")
        elif arm_id == 2:
            self.franka2_bodies = list(
                range(franka_body_start, template.body_count)
            )
            self.franka2_joints = list(
                range(franka_joint_start, template.joint_count)
            )
            self.franka2_shapes = list(
                range(franka_shape_start, template.shape_count)
            )
            self.gripper2_bodies = [
                body
                for body in self.franka2_bodies
                if "hand" in template.body_label[body]
                or "finger" in template.body_label[body]
            ]
            if not self.gripper2_bodies:
                raise RuntimeError("No gripper2 bodies found")

    def reset(self):
        """Restore initial spawn state."""
        self.model.joint_q.assign(self.init_joint_q)
        self.model.joint_qd.assign(self.init_joint_qd)
        for st in (self.state_0, self.state_1):
            newton.eval_fk(
                self.model, self.model.joint_q, self.model.joint_qd, st
            )
            st.body_qd.zero_()
            st.clear_forces()
        self.sim_time = 0.0

    def capture(self):
        """Warm up/preload GPU kernels and capture simulation substeps into a CUDA Graph."""
        self.use_graph = True
        # Preload / warm-up step to allocate internal buffers (e.g. SolverVBD contact history)
        self.simulate()
        self.reset()

        # Capture simulation graph
        with wp.ScopedDevice(self.device), wp.ScopedCapture() as capture:
            self.simulate()
        if capture.graph is None:
            raise RuntimeError(f"Graph capture failed on device {self.device}")
        self.graph = capture.graph
        self.reset()
        return self.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph is not None:
            with wp.ScopedDevice(self.device):
                wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    @staticmethod
    def _compute_lookat_transform(eye, target, up=(0.0, 0.0, 1.0)):
        eye = np.array(eye, dtype=np.float32)
        target = np.array(target, dtype=np.float32)
        up = np.array(up, dtype=np.float32)

        # OpenGL / USD camera looks along -Z
        forward = target - eye
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        cam_up = np.cross(right, forward)
        cam_up = cam_up / np.linalg.norm(cam_up)

        # World-to-camera matrix columns: [right, cam_up, -forward]
        R = np.column_stack([right, cam_up, -forward])
        import scipy.spatial.transform

        q = scipy.spatial.transform.Rotation.from_matrix(R).as_quat()
        return wp.transformf(wp.vec3f(*eye), wp.quatf(*q))

    def _setup_camera(self):
        """Initialize SensorTiledCamera, ray directions, and camera transforms."""
        self.camera_sensor = SensorTiledCamera(self.model)
        fov_rad = float(np.deg2rad(self.camera_fov))
        self.camera_rays = self.camera_sensor.utils.compute_camera_rays_pinhole(
            width=self.camera_width,
            height=self.camera_height,
            camera_fovs=fov_rad,
        )
        tf = self._compute_lookat_transform(
            eye=self.camera_pos,
            target=self.camera_target,
        )
        self.camera_transforms = wp.array(
            [[tf] * self.worlds_count],
            dtype=wp.transformf,
            device=self.model.device,
        )
        self._camera_color_output = (
            self.camera_sensor.utils.create_color_image_output(
                width=self.camera_width,
                height=self.camera_height,
                camera_count=1,
            )
        )
        self._camera_clear_data = SensorTiledCamera.ClearData(
            clear_color=0xFFE6EDF2,
            clear_depth=1000.0,
        )

    def render_camera(self, world_index: int = 0) -> np.ndarray:
        """Render and return the current camera frame as an (H, W, 3) uint8 RGB numpy array."""
        if not self.enable_camera:
            raise RuntimeError(
                "Camera sensor is not enabled. Initialize with enable_camera=True."
            )

        self.model.bvh_refit_shapes(self.state_0)
        self.camera_sensor.update(
            state=self.state_0,
            camera_transforms=self.camera_transforms,
            camera_rays=self.camera_rays,
            color_image=self._camera_color_output,
            clear_data=self._camera_clear_data,
        )
        rgba = self.camera_sensor.utils.to_rgba_from_color(
            self._camera_color_output
        )
        return rgba.numpy()[world_index, :, :, :3]

    def get_camera_image(self, world_index: int = 0):
        """Render and return the current camera frame as a PIL Image."""
        from PIL import Image

        rgb = self.render_camera(world_index=world_index)
        return Image.fromarray(rgb)

    def save_camera_frame(
        self, filepath: str | Path, world_index: int = 0, quality: int = 95
    ) -> Path:
        """Render and save the current camera frame to a JPEG or PNG file."""
        from PIL import Image

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        img = self.get_camera_image(world_index=world_index)
        img.save(str(path), quality=quality)
        return path

    @staticmethod
    def save_camera_animation(
        frames: list[np.ndarray], filepath: str | Path, fps: int = 30
    ) -> Path:
        """Save a list of RGB numpy frames as an animated GIF or video."""
        from PIL import Image

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not frames:
            raise ValueError("frames list cannot be empty")
        pil_images = [
            Image.fromarray(f) if isinstance(f, np.ndarray) else f
            for f in frames
        ]
        pil_images[0].save(
            str(path),
            save_all=True,
            append_images=pil_images[1:],
            duration=int(1000 / max(1, fps)),
            loop=0,
        )
        return path


if __name__ == "__main__":
    print("Hello World")
    dexm = DexmBuilder(use_graph=True, enable_camera=True)
    print("Graph captured:", dexm.graph is not None)
    for _ in range(10):
        dexm.step()
    print("Step successful, sim_time:", dexm.sim_time)
    frame = dexm.render_camera()
    print("Camera rendered frame shape:", frame.shape)
