"""
BI-SO100 DUAL ARM ROBOT IMPLEMENTATION
=====================================

This module implements the BI-SO100 robot agent for ManiSkill simulation.
The BI-SO100 robot is a stationary dual-arm manipulation platform featuring:

HARDWARE CONFIGURATION:
- Fixed mounting base (no mobility)
- Dual SO-100 robotic arms (5 DOF each + gripper)
- No head system or pan-tilt mechanisms
- Total: 12 degrees of freedom

KEY FEATURES:
- Dual-arm manipulation capabilities
- Hand-mounted cameras on each gripper
- Configurable control modes (position, velocity, force)
- Gripper force sensing and grasp detection

CONTROL MODES:
- pd_joint_pos: Direct joint position control
- pd_joint_delta_pos: Incremental joint position control
- pd_ee_delta_pos: End-effector position control
- pd_ee_delta_pose: End-effector pose control
- pd_joint_vel: Joint velocity control
- Dual-arm coordination modes

This is a stationary version derived from the Fetch robot, designed specifically
for tabletop dual-arm manipulation tasks.
"""

from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
import sapien
import sapien.physx as physx
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.link import Link
from mani_skill.utils.structs.types import Array


@register_agent()
class BiSO100(BaseAgent):
    """
    Bimanual SO-100 Robot Agent for Tabletop Manipulation
    
    A clean dual-arm manipulation robot optimized for tabletop tasks with:
    - 2x SO-100 arms (5 DOF each + gripper) 
    - Invisible mounting base (no visual clutter)
    - Multiple cameras per arm: hand-mounted + wrist-mounted
    - 12 total DOF (6 per arm)
    - Symmetric arm positioning for coordinated manipulation
    
    CAMERA CONFIGURATION:
    - hand_camera_1/2: Close-up gripper view for precise manipulation
    - wrist_camera_1/2: Wider workspace view for visual servoing and navigation

    Upper arm length: approximately 0.11257m
    Lower arm length: approximately 0.1349m
    Wrist extension: approximately 0.097m
    """
    uid = "bi_so100"
    urdf_path = f"bi_lerobot/assets/robots/bi_so100/bi_so100.urdf"
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            Fixed_Jaw=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            Moving_Jaw=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            Fixed_Jaw_2=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            Moving_Jaw_2=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )

    keyframes = dict(
        rest=Keyframe(
            pose=sapien.Pose(),
            # 12 DOF: 5 arm1 + 1 gripper1 + 5 arm2 + 1 gripper2
            # Using exact same arm joint values as fetch robot with correct joint mapping
            # [ 0.18780832  3.38986778  3.10115719  0.92896205 -0.08568161  0.03657684 
            # 0.01516847  3.40625191  3.14159083  0.94397527  0.07388389  0.68610877]

            qpos=np.array([
                # Joint order: Rotation, Rotation_2, Pitch, Pitch_2, Elbow, Elbow_2, Wrist_Pitch, Wrist_Pitch_2, Wrist_Roll, Wrist_Roll_2, Jaw, Jaw_2
                0,   # Rotation (arm1)
                0,   # Rotation_2 (arm2)  
                3.14, # Pitch (arm1)
                3.14, # Pitch_2 (arm2)
                3.14, # Elbow (arm1) 
                3.14, # Elbow_2 (arm2)
                0.9,   # Wrist_Pitch (arm1)
                0.9,   # Wrist_Pitch_2 (arm2)
                0,   # Wrist_Roll (arm1)
                0,   # Wrist_Roll_2 (arm2)
                0,   # Jaw (arm1)
                0    # Jaw_2 (arm2)
            ]),
        )
    )

    @property
    def _sensor_configs(self):
        """
        Configure wrist cameras mounted between jaw and wrist roll joint to match real SO-100 hardware
        
        Cameras are positioned on the gripper assembly pointing forward into the workspace.
        """
        return [
            # FIRST ARM CAMERA
            # Wrist camera mounted on right side of jaw using dedicated camera link
            CameraConfig(
                uid="wrist_camera_1", 
                pose=Pose.create_from_pq([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),  # Identity transform - camera link already positioned correctly
                width=640,
                height=480,
                fov=1.3,  # Wide field of view for workspace monitoring
                near=0.01,
                far=100,
                entity_uid="Left_Arm_Camera",  # Mount to dedicated camera link
            ),
            
            # SECOND ARM CAMERA  
            # Wrist camera mounted on right side of jaw using dedicated camera link
            CameraConfig(
                uid="wrist_camera_2",
                pose=Pose.create_from_pq([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),  # Identity transform - camera link already positioned correctly
                width=640,
                height=480,
                fov=1.3,  # Wide field of view for workspace monitoring
                near=0.01,
                far=100,
                entity_uid="Right_Arm_Camera",  # Mount to dedicated camera link
            ),
        ]

    def __init__(self, *args, **kwargs):
        """
        Initialize the bimanual SO-100 robot controller
        
        Defines joint configurations for both arms and grippers.
        No mobile base or head system.
        """
        # FIRST ARM CONFIGURATION
        self.arm_joint_names = [
            "Rotation",
            "Pitch", 
            "Elbow",
            "Wrist_Pitch",
            "Wrist_Roll",
        ]
        self.arm_stiffness = 2e4
        self.arm_damping = 1e2
        self.arm_force_limit = 250

        self.gripper_joint_names = [
            "Jaw",
        ]
        self.gripper_stiffness = 40
        self.gripper_damping = 1e2
        self.gripper_force_limit = 3

        self.ee_link_name = "Fixed_Jaw"
        
        # SECOND ARM CONFIGURATION  
        self.arm2_joint_names = [
            "Rotation_2",
            "Pitch_2",
            "Elbow_2", 
            "Wrist_Pitch_2",
            "Wrist_Roll_2",
        ]
        self.arm2_stiffness = 2e4
        self.arm2_damping = 1e2
        self.arm2_force_limit = 250

        self.gripper2_joint_names = [
            "Jaw_2",
        ]
        self.gripper2_stiffness = 40
        self.gripper2_damping = 1e2
        self.gripper2_force_limit = 3

        self.ee2_link_name = "Fixed_Jaw_2"

        super().__init__(*args, **kwargs)

    @property
    def _controller_configs(self):
        """
        Define controller configurations for the dual-arm system
        
        Provides multiple control modes for both arms independently
        and coordinated dual-arm control modes.
        """
        # FIRST ARM CONTROLLERS
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        # End-effector control for first arm
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )

        arm_pd_ee_target_delta_pos = deepcopy(arm_pd_ee_delta_pos)
        arm_pd_ee_target_delta_pos.use_target = True
        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        # End-effector control with human-friendly alignment (for teleoperation)
        arm_pd_ee_delta_pose_align = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_delta_pose_align.frame = "ee_align"

        # Joint velocity control: Direct velocity commands
        arm_pd_joint_vel = PDJointVelControllerConfig(
            self.arm_joint_names,
            -1.0,  # Max negative velocity
            1.0,   # Max positive velocity
            self.arm_damping,  # this might need to be tuned separately
            self.arm_force_limit,
        )

        # Combined position and velocity control
        arm_pd_joint_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=True,
        )
        arm_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )

        # FIRST ARM GRIPPER CONTROLLER
        gripper_pd_joint_pos = PDJointPosControllerConfig(
            self.gripper_joint_names,
            -20,   # closed position 
            20,   # open position (from URDF limit)
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
        )

        # SECOND ARM CONTROLLERS
        arm2_pd_joint_pos = PDJointPosControllerConfig(
            self.arm2_joint_names,
            None,
            None,
            self.arm2_stiffness,
            self.arm2_damping,
            self.arm2_force_limit,
            normalize_action=False,
        )
        arm2_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm2_joint_names,
            -0.1,
            0.1,
            self.arm2_stiffness,
            self.arm2_damping,
            self.arm2_force_limit,
            use_delta=True,
        )
        arm2_pd_joint_target_delta_pos = deepcopy(arm2_pd_joint_delta_pos)
        arm2_pd_joint_target_delta_pos.use_target = True

        # End-effector control for second arm
        arm2_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm2_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.arm2_stiffness,
            damping=self.arm2_damping,
            force_limit=self.arm2_force_limit,
            ee_link=self.ee2_link_name,
            urdf_path=self.urdf_path,
        )
        arm2_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm2_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm2_stiffness,
            damping=self.arm2_damping,
            force_limit=self.arm2_force_limit,
            ee_link=self.ee2_link_name,
            urdf_path=self.urdf_path,
        )

        arm2_pd_ee_target_delta_pos = deepcopy(arm2_pd_ee_delta_pos)
        arm2_pd_ee_target_delta_pos.use_target = True
        arm2_pd_ee_target_delta_pose = deepcopy(arm2_pd_ee_delta_pose)
        arm2_pd_ee_target_delta_pose.use_target = True

        arm2_pd_ee_delta_pose_align = deepcopy(arm2_pd_ee_delta_pose)
        arm2_pd_ee_delta_pose_align.frame = "ee_align"

        # Second arm velocity control
        arm2_pd_joint_vel = PDJointVelControllerConfig(
            self.arm2_joint_names,
            -1.0,
            1.0,
            self.arm2_damping,
            self.arm2_force_limit,
        )

        arm2_pd_joint_pos_vel = PDJointPosVelControllerConfig(
            self.arm2_joint_names,
            None,
            None,
            self.arm2_stiffness,
            self.arm2_damping,
            self.arm2_force_limit,
            normalize_action=True,
        )
        arm2_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
            self.arm2_joint_names,
            -0.1,
            0.1,
            self.arm2_stiffness,
            self.arm2_damping,
            self.arm2_force_limit,
            use_delta=True,
        )

        # SECOND ARM GRIPPER CONTROLLER
        gripper2_pd_joint_pos = PDJointPosControllerConfig(
            self.gripper2_joint_names,
            -20,   # closed position
            20,   # open position (from URDF limit)
            self.gripper2_stiffness,
            self.gripper2_damping,
            self.gripper2_force_limit,
        )

        # CONTROLLER CONFIGURATION DICTIONARY
        # Single arm control modes (for backward compatibility and individual arm control)
        controller_configs = dict(
            # Dual-arm coordinated control modes
            pd_joint_delta_pos_dual_arm=dict(
                arm1=arm_pd_joint_delta_pos,
                gripper1=gripper_pd_joint_pos,
                arm2=arm2_pd_joint_delta_pos,
                gripper2=gripper2_pd_joint_pos,
            ),
            
            pd_joint_pos_dual_arm=dict(
                arm1=arm_pd_joint_pos,
                gripper1=gripper_pd_joint_pos,
                arm2=arm2_pd_joint_pos,
                gripper2=gripper2_pd_joint_pos,
            ),
            
            pd_ee_delta_pos_dual_arm=dict(
                arm1=arm_pd_ee_delta_pos,
                gripper1=gripper_pd_joint_pos,
                arm2=arm2_pd_ee_delta_pos,
                gripper2=gripper2_pd_joint_pos,
            ),
            pd_ee_delta_pose_dual_arm=dict(
                arm1=arm_pd_ee_delta_pose,
                gripper1=gripper_pd_joint_pos,
                arm2=arm2_pd_ee_delta_pose,
                gripper2=gripper2_pd_joint_pos,
            ),

            # First arm individual control
            pd_joint_pos=dict(
                arm=arm_pd_joint_pos,
                gripper=gripper_pd_joint_pos,
            ),
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos,
                gripper=gripper_pd_joint_pos,
            ),
            pd_ee_delta_pos=dict(
                arm=arm_pd_ee_delta_pos,
                gripper=gripper_pd_joint_pos,
            ),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose,
                gripper=gripper_pd_joint_pos,
            ),
            pd_ee_delta_pose_align=dict(
                arm=arm_pd_ee_delta_pose_align,
                gripper=gripper_pd_joint_pos,
            ),
            pd_joint_target_delta_pos=dict(
                arm=arm_pd_joint_target_delta_pos,
                gripper=gripper_pd_joint_pos,
            ),
            pd_ee_target_delta_pos=dict(
                arm=arm_pd_ee_target_delta_pos,
                gripper=gripper_pd_joint_pos,
            ),
            pd_ee_target_delta_pose=dict(
                arm=arm_pd_ee_target_delta_pose,
                gripper=gripper_pd_joint_pos,
            ),
            pd_joint_vel=dict(
                arm=arm_pd_joint_vel,
                gripper=gripper_pd_joint_pos,
            ),
            pd_joint_pos_vel=dict(
                arm=arm_pd_joint_pos_vel,
                gripper=gripper_pd_joint_pos,
            ),
            pd_joint_delta_pos_vel=dict(
                arm=arm_pd_joint_delta_pos_vel,
                gripper=gripper_pd_joint_pos,
            ),
            
            # Second arm individual control
            pd_joint_pos_arm2=dict(
                arm=arm2_pd_joint_pos,
                gripper=gripper2_pd_joint_pos,
            ),
            pd_joint_delta_pos_arm2=dict(
                arm=arm2_pd_joint_delta_pos,
                gripper=gripper2_pd_joint_pos,
            ),
            pd_ee_delta_pos_arm2=dict(
                arm=arm2_pd_ee_delta_pos,
                gripper=gripper2_pd_joint_pos,
            ),
            pd_ee_delta_pose_arm2=dict(
                arm=arm2_pd_ee_delta_pose,
                gripper=gripper2_pd_joint_pos,
            ),
            pd_ee_delta_pose_align_arm2=dict(
                arm=arm2_pd_ee_delta_pose_align,
                gripper=gripper2_pd_joint_pos,
            ),
            pd_joint_target_delta_pos_arm2=dict(
                arm=arm2_pd_joint_target_delta_pos,
                gripper=gripper2_pd_joint_pos,
            ),
            pd_ee_target_delta_pos_arm2=dict(
                arm=arm2_pd_ee_target_delta_pos,
                gripper=gripper2_pd_joint_pos,
            ),
            pd_ee_target_delta_pose_arm2=dict(
                arm=arm2_pd_ee_target_delta_pose,
                gripper=gripper2_pd_joint_pos,
            ),
            pd_joint_vel_arm2=dict(
                arm=arm2_pd_joint_vel,
                gripper=gripper2_pd_joint_pos,
            ),
            pd_joint_pos_vel_arm2=dict(
                arm=arm2_pd_joint_pos_vel,
                gripper=gripper2_pd_joint_pos,
            ),
            pd_joint_delta_pos_vel_arm2=dict(
                arm=arm2_pd_joint_delta_pos_vel,
                gripper=gripper2_pd_joint_pos,
            ),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy(controller_configs)

    def _after_init(self):
        """
        Initialize robot link references after URDF loading
        
        Sets up references to important robot links for grasp detection
        and collision handling. Only arm and gripper links (no mobile base or head).
        """
        # FIRST ARM GRIPPER LINKS
        self.finger1_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "Fixed_Jaw"
        )
        self.finger2_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "Moving_Jaw"
        )
        self.finger1_tip: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "Fixed_Jaw_tip"
        )
        self.finger2_tip: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "Moving_Jaw_tip"
        )
        self.tcp: Link = self.finger1_link
        
        # SECOND ARM GRIPPER LINKS
        self.finger1_link_2: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "Fixed_Jaw_2"
        )
        self.finger2_link_2: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "Moving_Jaw_2"
        )
        self.finger1_tip_2: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "Fixed_Jaw_tip_2"
        )
        self.finger2_tip_2: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "Moving_Jaw_tip_2"
        )
        self.tcp_2: Link = self.finger1_link_2

        # BASE LINK (stationary mounting platform)
        self.base_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "base_link"
        )

        # Initialize contact query dictionary for grasp detection
        self.queries: Dict[
            str, Tuple[physx.PhysxGpuContactPairImpulseQuery, Tuple[int]]
        ] = dict()

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=110, arm_id=1):
        """
        Check if the robot is grasping an object with specified arm
        
        Args:
            object (Actor): The object to check if the robot is grasping
            min_force (float, optional): Minimum force before the robot is considered to be grasping the object in Newtons. Defaults to 0.5.
            max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 110.
            arm_id (int, optional): Which arm to check (1 or 2). Defaults to 1.
        
        Returns:
            torch.Tensor: Boolean tensor indicating if the robot is grasping the object
        """
        if arm_id == 1:
            finger1_link = self.finger1_link
            finger2_link = self.finger2_link
        elif arm_id == 2:
            finger1_link = self.finger1_link_2
            finger2_link = self.finger2_link_2
        else:
            raise ValueError(f"arm_id must be 1 or 2, got {arm_id}")

        l_contact_forces = self.scene.get_pairwise_contact_forces(
            finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        # direction to open the gripper
        ldirection = finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)

    def is_static(self, threshold=0.2):
        """
        Check if the robot is static (not moving)
        
        For the stationary robot, this only checks arm joint velocities.
        
        Args:
            threshold (float): Maximum joint velocity to consider static
            
        Returns:
            torch.Tensor: Boolean tensor indicating if robot is static
        """
        qvel = self.robot.get_qvel()  # Get all joint velocities (12 DOF for dual arm)
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        """
        Build a grasp pose for the gripper
        
        Args:
            approaching: Unit vector pointing in the direction the gripper approaches
            closing: Unit vector pointing in the direction the gripper closes  
            center: 3D position of the grasp center
            
        Returns:
            sapien.Pose: The grasp pose
        """
        # Validate input vectors are unit vectors and orthogonal
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return sapien.Pose(T)

    @property
    def tcp_pos(self):
        """Tool center point position for first arm (midpoint between gripper fingertips)"""
        return (self.finger1_tip.pose.p + self.finger2_tip.pose.p) / 2

    @property
    def tcp_pose(self):
        """Tool center point pose for first arm"""
        return Pose.create_from_pq(self.tcp_pos, self.finger1_link.pose.q)

    @property
    def tcp_pos_2(self):
        """Tool center point position for second arm (midpoint between gripper fingertips)"""
        return (self.finger1_tip_2.pose.p + self.finger2_tip_2.pose.p) / 2

    @property
    def tcp_pose_2(self):
        """Tool center point pose for second arm"""
        return Pose.create_from_pq(self.tcp_pos_2, self.finger1_link_2.pose.q)
