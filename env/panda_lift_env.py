"""
PandaLift — robosuite Lift task using the Panda robot.

Switched from SO-101 because the robosuite SO-101 gripper model (single
revolute jaw with no proper opposing collision geometry) cannot reliably
grasp objects with real contact physics. Panda has a well-tested parallel-jaw
gripper that actually works.

Differences from base Lift:
  - Constrained cube placement (tight, deterministic for demo collection)
  - Stable cube init (placed directly at rest, no bouncy drop)
  - Arm initialized to a safe home pose above the cube
  - Absolute lift success threshold (table_top + 5cm)
"""

import numpy as np

from robosuite.environments.manipulation.lift import Lift
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.mjcf_utils import CustomMaterial


# 40mm cube (same as default Lift)
CUBE_HALFSIZE_MIN = [0.020, 0.020, 0.020]
CUBE_HALFSIZE_MAX = [0.022, 0.022, 0.022]

# Cube must rise this many metres above the table top
LIFT_ABS_THRESHOLD = 0.05

# Panda home pose (reachable, clear of table)
PANDA_HOME_JOINTS = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 1.9, 0.785])


class PandaLift(Lift):
    """Panda Lift task with stable cube init and tight placement for demos."""

    def _load_model(self):
        super(Lift, self)._load_model()

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
            size_min=CUBE_HALFSIZE_MIN,
            size_max=CUBE_HALFSIZE_MAX,
            rgba=[1, 0, 0, 1],
            material=redwood,
            friction=[1.0, 0.05, 0.001],
        )

        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cube)
        else:
            # Tight placement right in front of Panda
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube,
                x_range=[-0.03, 0.03],
                y_range=[-0.03, 0.03],
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

    def _reset_internal(self):
        super()._reset_internal()
        self.sim.forward()

        cube_jnt_addr = self.sim.model.jnt_qposadr[
            self.sim.model.joint_name2id("cube_joint0")
        ]
        cube_jnt_dofadr = self.sim.model.jnt_dofadr[
            self.sim.model.joint_name2id("cube_joint0")
        ]

        # Place cube directly at rest (no bouncy drop)
        self._table_top_z = float(self.model.mujoco_arena.table_offset[2])
        cube_half_z = float(self.sim.model.geom_size[
            self.sim.model.geom_name2id("cube_g0")
        ][2])
        rest_z = self._table_top_z + cube_half_z + 0.002
        self.sim.data.qpos[cube_jnt_addr + 2] = rest_z
        self.sim.data.qvel[cube_jnt_dofadr:cube_jnt_dofadr + 6] = 0.0

        # Move arm to home pose
        try:
            self.robots[0].set_robot_joint_positions(PANDA_HOME_JOINTS)
        except Exception:
            pass

        self.sim.forward()
        self._cube_rest_z = rest_z
        self._success_abs_z = self._table_top_z + LIFT_ABS_THRESHOLD

    def _check_success(self):
        cube_z = float(self.sim.data.body_xpos[self.cube_body_id][2])
        thresh = getattr(self, "_success_abs_z", None)
        if thresh is None:
            thresh = 0.8 + LIFT_ABS_THRESHOLD
        return cube_z > thresh
