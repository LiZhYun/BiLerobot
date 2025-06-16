"""
Code for a minimal environment/task with just a robot being loaded. We recommend copying this template and modifying as you need.

At a high-level, ManiSkill tasks can minimally be defined by how the environment resets, what agents/objects are
loaded, goal parameterization, and success conditions

Environment reset is comprised of running two functions, `self._reconfigure` and `self.initialize_episode`, which is auto
run by ManiSkill. As a user, you can override a number of functions that affect reconfiguration and episode initialization.

Reconfiguration will reset the entire environment scene and allow you to load/swap assets and agents.

Episode initialization will reset the positions of all objects (called actors), articulations, and agents,
in addition to initializing any task relevant data like a goal

See comments for how to make your own environment and what each required function should do
"""

from typing import Any, Dict, Union

import numpy as np
import sapien
import torch
import torch.random
from transforms3d.euler import euler2quat
from bi_lerobot.agents.robots.bi_so100 import BiSO100

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.building import articulations
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig
from mani_skill import ASSET_DIR


@register_env("BiSO100OpenLid-v1", max_episode_steps=50)
class BiSO100OpenLidEnv(BaseEnv):
    """
    **Task Description:**
    A simple task where the objective is to open the lid of a bottle.

    **Randomizations:**
    - the bottle is placed flat on the table
    - the lid is placed on the bottle

    **Success Conditions:**
    - the cube's xy position is within goal_radius (default 0.1) of the target's xy position by euclidean distance and the cube is still on the table.
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PushCube-v1_rt.mp4"

    SUPPORTED_ROBOTS = ["bi_so100"]

    # Specify some supported robot types
    agent: Union[BiSO100]

    # set some commonly used values
    goal_radius = 0.1
    cube_half_size = 0.02

    def __init__(self, *args, robot_uids="bi_so100", robot_init_qpos_noise=0.02, **kwargs):
        # specifying robot_uids="panda" as the default means gym.make("PushCube-v1") will default to using the panda arm.
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # Specify default simulation/gpu memory configurations to override any default values
    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        # registers two scene cameras: a top camera watch down the scene through upper side of robot and a side camera watch down the scene through the left side of robot
        
        # Top camera positioned on the opposite side of the robot (robot is at [-0.585, 0, -0.1])
        # Position the camera above and in front of the robot to watch down on the workspace
        # Use custom up vector to control camera orientation when looking straight down
        top_camera_pose = sapien_utils.look_at(eye=[-0.45, 0, 0.7], target=[-0.45, 0, 0], up=[1, 0, 0])
        top_camera_config = CameraConfig(
            "top_camera",
            top_camera_pose,
            640,
            480,
            1,
            0.01,
            100,
        )
        
        # Side camera positioned on the left side of the robot
        # Position the camera to the left and slightly elevated to watch the workspace from the side
        side_camera_pose = sapien_utils.look_at(eye=[-0.5, 0.56, 0.3], target=[-0.5, 0, 0.1])
        side_camera_config = CameraConfig(
            "side_camera",
            side_camera_pose,
            640,
            480,
            1,
            0.01,
            100,
        )
        
        return [top_camera_config, side_camera_config]
    @property
    def _default_human_render_camera_configs(self):
        # registers two scene cameras: a top camera watch down the scene through upper side of robot and a side camera watch down the scene through the left side of robot
        
        # Top camera positioned on the opposite side of the robot (robot is at [-0.585, 0, -0.1])
        # Position the camera above and in front of the robot to watch down on the workspace
        # Use custom up vector to control camera orientation when looking straight down
        top_camera_pose = sapien_utils.look_at(eye=[-0.45, 0, 0.7], target=[-0.45, 0, 0], up=[1, 0, 0])
        top_camera_config = CameraConfig(
            "top_camera",
            top_camera_pose,
            640,
            480,
            1,
            0.01,
            100,
        )
        
        # Side camera positioned on the left side of the robot
        # Position the camera to the left and slightly elevated to watch the workspace from the side
        side_camera_pose = sapien_utils.look_at(eye=[-0.5, 0.56, 0.3], target=[-0.5, 0, 0.1])
        side_camera_config = CameraConfig(
            "side_camera",
            side_camera_pose,
            640,
            480,
            1,
            0.01,
            100,
        )
        
        return [top_camera_config, side_camera_config]

    def _load_agent(self, options: dict):
        # set a reasonable initial pose for the agent that doesn't intersect other objects
        super()._load_agent(options, sapien.Pose(p=[-0.585, 0, -0.1]))

    def _load_scene(self, options: dict):
        # we use a prebuilt scene builder class that automatically loads in a floor and table.
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # 🔧 SET SCALE FOR YCB CUP - CLEAN APPROACH
        # Method 1: Create custom scaled YCB builder
        self.mug = self._build_scaled_ycb_cup(scale=1.0)
        
        # Method 2: Alternative - modify existing builder (commented out)
        # builder = actors.get_actor_builder(
        #     self.scene,
        #     id=f"ycb:065-a_cups",
        # )
        # # Modify existing visual shapes scale
        # for visual_record in builder.visual_records:
        #     if hasattr(visual_record, 'scale'):
        #         visual_record.scale = [custom_scale] * 3
        # # Modify existing collision shapes scale  
        # for collision_record in builder.collision_records:
        #     if hasattr(collision_record, 'scale'):
        #         collision_record.scale = [custom_scale] * 3
        # builder.initial_pose = sapien.Pose(p=[-0.4, -0.15, 0.02], q=[0, 0, 0, 1])
        # self.mug = builder.build(name="mug")

        # 🔧 LOAD BOTTLE WITH CUSTOM SCALE
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = False
        # Method 1: Set scale directly on the loader (recommended for URDF)
        loader.scale = 0.12  # Scale to 12% of original size
        loader.load_multiple_collisions_from_file = True
        
        urdf_path = "/home/zhiyuan/.maniskill/data/partnet_mobility/dataset/3822/mobility.urdf"
        applied_urdf_config = sapien_utils.parse_urdf_config(
            dict(
                material=dict(static_friction=1, dynamic_friction=1, restitution=0),
                # Method 2: Alternative - you can also set scale in URDF config
                # scale=0.12  # This would override loader.scale
            )
        )
        # applied_urdf_config.update(**urdf_config)
        sapien_utils.apply_urdf_config(loader, applied_urdf_config)
        articulation_builders = loader.parse(str(urdf_path))["articulation_builders"]
        builder2 = articulation_builders[0]
        builder2.initial_pose = sapien.Pose(p=[-0.4, 0.15, 0.05], q=[1, 0, 0, 0])
        self.bottle = builder2.build(name="bottle")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Randomize mug position slightly around its initial position
            mug_xyz = torch.zeros((b, 3))
            mug_xyz[:, 0] = -0.4 + torch.rand((b,)) * 0.1 - 0.05  # x: -0.45 to -0.35
            mug_xyz[:, 1] = -0.15 + torch.rand((b,)) * 0.1 - 0.05  # y: -0.2 to -0.1
            mug_xyz[:, 2] = 0.02  # keep on table surface
            mug_pose = Pose.create_from_pq(mug_xyz, torch.tensor([0, 0, 0, 1]).repeat(b, 1))
            
            # Randomize bottle position slightly around its initial position  
            bottle_xyz = torch.zeros((b, 3))
            bottle_xyz[:, 0] = -0.4 + torch.rand((b,)) * 0.1 - 0.05  # x: -0.45 to -0.35
            bottle_xyz[:, 1] = 0.15 + torch.rand((b,)) * 0.1 - 0.05  # y: 0.1 to 0.2
            bottle_xyz[:, 2] = 0.1  # keep on table surface
            bottle_pose = Pose.create_from_pq(bottle_xyz, torch.tensor([1, 0, 0, 0]).repeat(b, 1))
            
            # Apply poses to objects
            self.mug.set_pose(mug_pose)
            self.bottle.set_pose(bottle_pose)
            
            # # Randomize mug position slightly around its initial position
            # mug_xyz = torch.zeros((b, 3))
            # mug_xyz[:, 0] = 0.1 + torch.rand((b,)) * 0.1 - 0.05  # x: 0.05 to 0.15
            # mug_xyz[:, 1] = -0.2 + torch.rand((b,)) * 0.1 - 0.05  # y: -0.25 to -0.15
            # mug_xyz[:, 2] = 0.05  # keep on table surface
            # mug_pose = Pose.create_from_pq(mug_xyz, torch.tensor([1, 0, 0, 0]).repeat(b, 1))
            
            # # Randomize bottle position slightly around its initial position  
            # bottle_xyz = torch.zeros((b, 3))
            # bottle_xyz[:, 0] = 0.1 + torch.rand((b,)) * 0.1 - 0.05  # x: 0.05 to 0.15
            # bottle_xyz[:, 1] = 0.2 + torch.rand((b,)) * 0.1 - 0.05  # y: 0.15 to 0.25
            # bottle_xyz[:, 2] = 0.05  # keep on table surface
            # bottle_pose = Pose.create_from_pq(bottle_xyz, torch.tensor([1, 0, 0, 0]).repeat(b, 1))
            
            # # Apply poses to objects
            # self.mug.set_pose(mug_pose)
            # self.bottle.set_pose(bottle_pose)

    # def evaluate(self):
    #     return {
    #         "success": True,
    #     }

    def _get_obs_extra(self, info: Dict):
        # some useful observation info for solving the task includes the pose of the tcp (tool center point) which is the point between the
        # grippers of the robot
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        return obs
    
    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        reward = 0.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 4.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward

    def _build_scaled_ycb_cup(self, scale=1.0):
        """
        Build a YCB cup with custom scaling
        """
        from mani_skill import ASSET_DIR
        from mani_skill.utils.io_utils import load_json
        
        # Load YCB dataset info
        model_data = load_json(ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json")
        cup_id = "065-b_cups"
        metadata = model_data[cup_id]
        
        # Create builder and add shapes with custom scale
        builder = self.scene.create_actor_builder()
        density = metadata.get("density", 1000)
        model_dir = ASSET_DIR / "assets/mani_skill2_ycb/models" / cup_id
        
        # Add collision with custom scale
        collision_file = str(model_dir / "collision.ply")
        builder.add_multiple_convex_collisions_from_file(
            filename=collision_file,
            scale=[scale] * 3,
            density=density,
        )
        
        # Add visual with custom scale
        visual_file = str(model_dir / "textured.obj")
        builder.add_visual_from_file(filename=visual_file, scale=[scale] * 3)
        
        # Set pose and build
        builder.initial_pose = sapien.Pose(p=[-0.4, -0.15, 0.02], q=[0, 0, 0, 1])
        return builder.build(name="mug")

if __name__ == "__main__":
    import gymnasium as gym
    env: BaseEnv = gym.make(
        "BiSO100OpenLid-v1",
        render_mode="human",
    )
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    # Test for a few episodes
    num_episodes = 2
    for i in range(num_episodes):
        print(f"\n--- Starting Episode {i+1} ---")
        obs, info = env.reset() # Reset the environment and get initial observation and info
        
        # Run for a maximum number of steps per episode
        max_steps_per_episode = 100
        for step in range(max_steps_per_episode):
            # action = env.action_space.sample() # Take a random action
            # obs, reward, terminated, truncated, info = env.step(action) # Step the environment
            
            # Print relevant information
            # print(f"Step {step}: Reward={reward:.4f}, Terminated={terminated}, Truncated={truncated}")
            # You can also inspect `obs` or `info` for more details, e.g., info["success"]
            
            env.render() # Render the scene (if render_mode="human")
    
    env.close() # Close the environment when done with testing
    print("\n--- Testing complete ---")