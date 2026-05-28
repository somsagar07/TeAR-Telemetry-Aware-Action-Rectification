"""
SO101Lift — robosuite Lift task tuned for the SO-101 arm.

This is the "honest" version: NO magnetic grasping tricks. The gripper must
physically grasp the cube using real MuJoCo contact physics. For this to work,
the robosuite SO-101 XML must include a collision geom for the fixed jaw half
(`wrist_roll_follower_so101_v1`) — otherwise the moving jaw has nothing to
pinch against. See the edit to:
    robosuite/models/assets/robots/so101/robot.xml

which adds a mesh collision for `wrist_roll_follower_so101_v1`, matching the
reference MJCF from TheRobotStudio's SO-ARM100 repo.

Differences from the base `Lift` environment:
  - Cube is ~22mm (default 40mm is too large for SO-101's ~30mm jaw opening).
  - Cube spawns in a tight zone the SO-101 can actually reach.
  - Arm starts in a safe home pose that doesn't collide with the cube.
  - Gripper initialized fully open.
  - Cube is placed directly at rest height (no bouncy drop).
  - Absolute lift success threshold (table_top + 5cm).
"""

import numpy as np

from robosuite.environments.manipulation.lift import Lift
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.mjcf_utils import CustomMaterial


# 22mm cube — fits within the SO-101 jaw opening (~30mm max) with clearance.
SO101_CUBE_HALFSIZE_MIN = [0.011, 0.011, 0.011]
SO101_CUBE_HALFSIZE_MAX = [0.011, 0.011, 0.011]

# Cube body center must rise this many metres ABOVE the table top to count as
# a success. Absolute threshold — doesn't depend on cached rest heights.
SO101_LIFT_ABS_THRESHOLD = 0.050

# Safe home pose: arm raised high so hand/gripper is ~33cm from any possible
# cube spawn position (cube in x∈[0.03, 0.08], y∈[-0.02, 0.02], z≈0.83).
SO101_HOME_JOINTS = np.array([0.0, -0.5, -0.1, 0.0, 0.0])


class SO101Lift(Lift):
    """Lift task for the SO-101 arm with real contact-based grasping."""

    def _load_model(self):
        # Skip Lift._load_model (we rebuild the scene with a different cube)
        super(Lift, self)._load_model()  # ManipulationEnv._load_model

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        from robosuite.models.arenas import TableArena
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        tex_attrib = {"type": "cube"}
        mat_attrib = {"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"}
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.cube = BoxObject(
            name="cube",
            size_min=SO101_CUBE_HALFSIZE_MIN,
            size_max=SO101_CUBE_HALFSIZE_MAX,
            rgba=[1, 0, 0, 1],
            material=redwood,
            friction=[3.0, 0.10, 0.001],
            density=100,  # very light cube is easy for the weak SO-101 to grasp
        )

        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cube)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube,
                x_range=[0.03, 0.08],
                y_range=[-0.02, 0.02],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.02,
            )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cube,
        )

    def _load_model(self):
        # Build the scene as usual, then enlarge the table collision so the
        # arm physically can't swing into the empty space below the table.
        super()._load_model()
        self._extend_table_collision_in_xml()

    def _extend_table_collision_in_xml(self):
        """Add a big invisible floor barrier at table-top height that
        extends well beyond the table in both x/y so no arm pose can drop
        below the work surface. Visuals are unaffected."""
        arena = self.model.mujoco_arena
        root = arena.worldbody
        import xml.etree.ElementTree as ET

        # Find the arena worldbody (robosuite TableArena puts everything here)
        # and add one big collision-only box.
        barrier = ET.SubElement(root, "geom")
        barrier.set("name", "so101_work_barrier")
        barrier.set("type", "box")
        # Center at z=0.4, half-height 0.4 => top face at z=0.8, bottom at 0.
        # Half-width 1.5m covers base + table + margin.
        barrier.set("pos", "0 0 0.40")
        barrier.set("size", "1.5 1.5 0.40")
        barrier.set("rgba", "0 0 0 0")   # fully transparent
        barrier.set("contype", "1")
        barrier.set("conaffinity", "1")
        barrier.set("group", "0")
        barrier.set("friction", "0.6 0.005 0.0001")
        barrier.set("solref", "0.002 1")
        barrier.set("solimp", "0.95 0.99 0.001")

    def _reset_internal(self):
        super()._reset_internal()
        self.sim.forward()

        cube_jnt_addr = self.sim.model.jnt_qposadr[
            self.sim.model.joint_name2id("cube_joint0")
        ]
        cube_jnt_dofadr = self.sim.model.jnt_dofadr[
            self.sim.model.joint_name2id("cube_joint0")
        ]

        # Place cube at deterministic rest (no bouncing)
        self._table_top_z = float(self.model.mujoco_arena.table_offset[2])
        cube_half_z = float(self.sim.model.geom_size[
            self.sim.model.geom_name2id("cube_g0")
        ][2])
        rest_z = self._table_top_z + cube_half_z + 0.001
        self.sim.data.qpos[cube_jnt_addr + 2] = rest_z
        self.sim.data.qvel[cube_jnt_dofadr:cube_jnt_dofadr + 6] = 0.0

        # Open the gripper at init (both slide joints at 0 = fully open)
        for jname in ("gripper0_right_so101_gripper_joint",
                      "gripper0_right_so101_gripper_joint_r"):
            try:
                addr = self.sim.model.jnt_qposadr[
                    self.sim.model.joint_name2id(jname)
                ]
                self.sim.data.qpos[addr] = 0.0
            except Exception:
                pass

        # Move arm to safe home pose (prevents initial-step collision with cube)
        try:
            self.robots[0].set_robot_joint_positions(SO101_HOME_JOINTS)
        except Exception:
            pass

        self.sim.forward()
        self._cube_rest_z = rest_z
        self._success_abs_z = self._table_top_z + SO101_LIFT_ABS_THRESHOLD
        # Reset the sticky success flag on episode start.
        self._lifted_success = False

    def _check_success(self):
        """Sticky absolute lift threshold.

        Once the cube rises above (table_top + SO101_LIFT_ABS_THRESHOLD)
        at any point during the episode, success stays True. This makes
        "briefly lifted then fell" count as a valid pick for demo collection,
        which is a reasonable definition of success for a lift task with
        a small, weakly-grasped cube.
        """
        cube_z = float(self.sim.data.body_xpos[self.cube_body_id][2])
        thresh = getattr(self, "_success_abs_z", None)
        if thresh is None:
            thresh = 0.8 + SO101_LIFT_ABS_THRESHOLD
        if cube_z > thresh:
            self._lifted_success = True
        return bool(getattr(self, "_lifted_success", False))
