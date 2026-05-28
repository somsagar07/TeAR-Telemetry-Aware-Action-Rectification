"""
SO-101 Parallel Gripper — modified SO-101 gripper with a fixed opposing finger
so the moving jaw can actually grasp objects in physics.

The original SO-101 gripper has only one moving jaw with no opposing surface,
making it incapable of physically gripping a cube. This class loads a custom
XML that adds a fixed finger at a position where the closing jaw can sandwich
small objects.
"""

import os
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_XML_PATH = os.path.join(_THIS_DIR, "so101_parallel_gripper.xml")


class SO101ParallelGripper(GripperModel):
    """SO-101 with an extra fixed opposing finger added to the gripper XML."""

    def __init__(self, idn=0):
        super().__init__(_XML_PATH, idn=idn)

    def format_action(self, action):
        assert len(action) == self.dof
        self.current_action = np.clip(
            self.current_action + self.speed * np.sign(action),
            -1.0, 1.0,
        )
        return self.current_action

    @property
    def speed(self):
        return 0.2

    @property
    def dof(self):
        return 1

    @property
    def init_qpos(self):
        return np.array([0.0])

    @property
    def _important_geoms(self):
        return {
            "left_finger": ["jaw_collision"],
            "right_finger": ["fixed_finger_collision"],
            "left_fingerpad": ["jaw_pad_collision"],
            "right_fingerpad": ["fixed_finger_collision"],
        }
