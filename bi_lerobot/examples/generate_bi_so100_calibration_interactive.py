#!/usr/bin/env python3
"""
Interactive BiSO100 Calibration Generator

This script allows you to manually position both the physical leader arm and the virtual arm
to the same poses, then generates accurate calibration data based on these aligned positions.

This approach provides much better calibration than environment-based estimation because
it uses real-world correspondence between leader and virtual arm positions.

Usage:
    python examples/generate_bi_so100_calibration_interactive.py \
        --leader-id=right_leader \
        --virtual-arm=right \
        --teleop-port=/dev/ttyACM1

Controls:
    - Physical leader arm: Move manually to desired positions
    - Virtual arm: Use keyboard controls (same as demo_bi_so100_ctrl.py)
    - 'SPACE': Record current position correspondence
    - 'C': Calculate and save calibration
    - 'Q': Quit
"""

import json
import os
import argparse
import numpy as np
import gymnasium as gym
import pygame
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

import bi_lerobot  # Import to register ManiSkill environments

# Import LeRobot teleoperator
from lerobot.common.teleoperators.so100_leader import SO100LeaderConfig
from lerobot.common.teleoperators import make_teleoperator_from_config


@dataclass
class CalibrationPoint:
    """A single calibration point with leader and virtual arm positions"""
    leader_positions: Dict[str, float]  # In degrees from SO100
    virtual_positions: Dict[str, float]  # In radians from virtual arm
    description: str = ""


class InteractiveBiSO100Calibrator:
    """Interactive calibration generator using manual positioning"""
    
    def __init__(self, leader_id: str, virtual_arm: str, teleop_port: str, env_id: str = "BiSO100OpenLid-v1"):
        self.leader_id = leader_id
        self.virtual_arm = virtual_arm
        self.teleop_port = teleop_port
        self.env_id = env_id
        
        # Joint mappings
        self.joint_mappings = {
            "shoulder_pan": "Rotation",
            "shoulder_lift": "Pitch", 
            "elbow_flex": "Elbow",
            "wrist_flex": "Wrist_Pitch",
            "wrist_roll": "Wrist_Roll",
            "gripper": "Jaw",
        }
        
        # Adjust for virtual arm (add _2 suffix for right arm)
        if virtual_arm == "right":
            self.joint_mappings = {
                leader_joint: f"{virtual_joint}_2" 
                for leader_joint, virtual_joint in self.joint_mappings.items()
            }
        
        # Virtual joint indices in bi_so100 action space
        if virtual_arm == "left":
            self.virtual_indices = {
                "Rotation": 0, "Pitch": 1, "Elbow": 2,
                "Wrist_Pitch": 3, "Wrist_Roll": 4, "Jaw": 5
            }
        else:  # right
            self.virtual_indices = {
                "Rotation_2": 6, "Pitch_2": 7, "Elbow_2": 8,
                "Wrist_Pitch_2": 9, "Wrist_Roll_2": 10, "Jaw_2": 11
            }
        
        # Calibration data storage
        self.calibration_points: List[CalibrationPoint] = []
        
        # Environment and teleoperator
        self.env = None
        self.robot = None
        self.teleop = None
        
        # Virtual arm control (from demo_bi_so100_ctrl.py)
        self.target_joints = None
        self.joint_step = 0.01
        self.p_gain = None
        
    def _get_mapped_joints(self, robot):
        """
        Get the current joint positions from the robot and map them correctly to the target joints.
        (Copied from demo_bi_so100_ctrl.py)
        """
        if robot is None:
            return np.zeros(12)  # Default size for action
            
        # Get full joint positions
        full_joints = robot.get_qpos()
        
        # Convert tensor to numpy array if needed
        if hasattr(full_joints, 'numpy'):
            full_joints = full_joints.numpy()
        
        # Handle case where it's a 2D tensor/array
        if full_joints.ndim > 1:
            full_joints = full_joints.squeeze()
        
        # Create the mapped joints array with correct size
        mapped_joints = np.zeros(12)
        
        # Map the joints according to the specified mapping
        if len(full_joints) >= 12:
            # First arm: [0,2,4,6,8,10] → [0,1,2,3,4,5]
            mapped_joints[0] = full_joints[0]
            mapped_joints[1] = full_joints[2]
            mapped_joints[2] = full_joints[4]
            mapped_joints[3] = full_joints[6]
            mapped_joints[4] = full_joints[8]
            mapped_joints[5] = full_joints[10]
            
            # Second arm: [1,3,5,7,9,11] → [6,7,8,9,10,11]
            mapped_joints[6] = full_joints[1]
            mapped_joints[7] = full_joints[3]
            mapped_joints[8] = full_joints[5]
            mapped_joints[9] = full_joints[7]
            mapped_joints[10] = full_joints[9]
            mapped_joints[11] = full_joints[11]
        
        return mapped_joints

    def setup_environment(self):
        """Setup the ManiSkill environment"""
        print(f"Loading environment: {self.env_id}")
        
        # Create environment
        self.env = gym.make(
            self.env_id,
            obs_mode="none",
            render_mode="human",
            sim_backend="auto",
            robot_uids="bi_so100"
        )
        
        # Reset environment
        obs, _ = self.env.reset()
        
        # Get robot instance
        if hasattr(self.env.unwrapped, "agent"):
            self.robot = self.env.unwrapped.agent.robot
        elif hasattr(self.env.unwrapped, "agents") and len(self.env.unwrapped.agents) > 0:
            self.robot = self.env.unwrapped.agents[0]
        else:
            raise RuntimeError("Could not find robot in environment")
        
        print(f"Robot loaded: {type(self.robot)}")
        
        # Initialize virtual arm control
        self.target_joints = np.zeros(12)
        self.p_gain = np.ones(12)
        self.p_gain[0:5] = 1.0   # First arm joints
        self.p_gain[5] = 0.04   # First arm gripper
        self.p_gain[6:11] = 1.0  # Second arm joints
        self.p_gain[11] = 0.04  # Second arm gripper
        
        # Set initial target positions
        current_joints = self._get_mapped_joints(self.robot)
        self.target_joints = current_joints.copy()

    def setup_teleoperator(self):
        """Setup the SO100 leader teleoperator"""
        print(f"Connecting to SO100 leader: {self.leader_id} on {self.teleop_port}")
        
        teleop_config = SO100LeaderConfig(
            port=self.teleop_port,
            id=self.leader_id,
        )
        
        self.teleop = make_teleoperator_from_config(teleop_config)
        self.teleop.connect()
        
        print("SO100 leader connected successfully")

    def get_leader_positions(self) -> Dict[str, float]:
        """Get current leader arm positions"""
        leader_action = self.teleop.get_action()
        
        # Extract positions (all in degrees from SO100)
        leader_positions = {}
        for joint_name in self.joint_mappings.keys():
            if joint_name in leader_action:
                leader_positions[joint_name] = leader_action[joint_name]
            elif f"{joint_name}.pos" in leader_action:
                leader_positions[joint_name] = leader_action[f"{joint_name}.pos"]
            else:
                leader_positions[joint_name] = 0.0
        
        return leader_positions

    def get_virtual_positions(self) -> Dict[str, float]:
        """Get current virtual arm positions"""
        current_joints = self._get_mapped_joints(self.robot)
        
        # Extract positions for the target virtual arm
        virtual_positions = {}
        for leader_joint, virtual_joint in self.joint_mappings.items():
            idx = self.virtual_indices[virtual_joint]
            virtual_positions[leader_joint] = current_joints[idx]
            
        return virtual_positions

    def record_calibration_point(self, description: str = ""):
        """Record current positions as a calibration point"""
        leader_pos = self.get_leader_positions()
        virtual_pos = self.get_virtual_positions()
        
        point = CalibrationPoint(
            leader_positions=leader_pos,
            virtual_positions=virtual_pos,
            description=description
        )
        
        self.calibration_points.append(point)
        
        print(f"\n📍 Recorded calibration point #{len(self.calibration_points)}: {description}")
        print("Leader positions (degrees):")
        for joint, pos in leader_pos.items():
            print(f"  {joint}: {pos:.2f}°")
        print("Virtual positions (radians):")
        for joint, pos in virtual_pos.items():
            print(f"  {joint}: {pos:.3f} rad ({np.degrees(pos):.1f}°)")

    def calculate_calibration(self) -> Dict[str, Any]:
        """Calculate calibration data from recorded points"""
        if len(self.calibration_points) < 2:
            raise ValueError("Need at least 2 calibration points")
        
        print(f"\nCalculating calibration from {len(self.calibration_points)} points...")
        
        calibration_data = {}
        
        for leader_joint in self.joint_mappings.keys():
            # Extract data for this joint
            leader_positions = []
            virtual_positions = []
            
            for point in self.calibration_points:
                # Convert leader position from degrees to radians for calculation
                leader_deg = point.leader_positions[leader_joint]
                leader_rad = np.radians(leader_deg)
                leader_positions.append(leader_rad)
                
                virtual_rad = point.virtual_positions[leader_joint]
                virtual_positions.append(virtual_rad)
            
            leader_positions = np.array(leader_positions)
            virtual_positions = np.array(virtual_positions)
            
            # Calculate ranges
            leader_min = np.min(leader_positions)
            leader_max = np.max(leader_positions)
            virtual_min = np.min(virtual_positions)
            virtual_max = np.max(virtual_positions)
            
            # Calculate linear mapping: virtual = scale * leader + offset
            # Using least squares fit
            if len(leader_positions) >= 2:
                # Fit line: virtual = scale * leader + offset
                A = np.vstack([leader_positions, np.ones(len(leader_positions))]).T
                scale_factor, offset = np.linalg.lstsq(A, virtual_positions, rcond=None)[0]
            else:
                # Single point - assume direct mapping
                scale_factor = 1.0
                offset = virtual_positions[0] - leader_positions[0]
            
            # Calculate home positions (center of range)
            leader_home = (leader_min + leader_max) / 2
            virtual_home = (virtual_min + virtual_max) / 2
            
            # Create calibration entry
            virtual_joint = self.joint_mappings[leader_joint]
            calibration_data[leader_joint] = {
                "leader_home": float(leader_home),
                "virtual_home": float(virtual_home),
                "leader_range_min": float(leader_min),
                "leader_range_max": float(leader_max),
                "virtual_range_min": float(virtual_min),
                "virtual_range_max": float(virtual_max),
                "scale_factor": float(scale_factor),
                "offset": float(offset),
                "virtual_joint": virtual_joint,
                "virtual_joint_index": self.virtual_indices[virtual_joint],
                "samples_count": len(self.calibration_points),
                "source": "interactive_manual_positioning",
                "leader_source": "so100_hardware",
                "virtual_source": "manual_positioning"
            }
            
            print(f"{leader_joint} → {virtual_joint}:")
            print(f"  Leader range: [{np.degrees(leader_min):.1f}°, {np.degrees(leader_max):.1f}°]")
            print(f"  Virtual range: [{np.degrees(virtual_min):.1f}°, {np.degrees(virtual_max):.1f}°]")
            print(f"  Scale: {scale_factor:.4f}, Offset: {offset:.4f}")
        
        return calibration_data

    def save_calibration(self, calibration_data: Dict[str, Any], output_dir: str = None) -> bool:
        """Save calibration data to file"""
        try:
            # Default output directory following LeRobot convention
            if output_dir is None:
                output_dir = Path.home() / ".cache/huggingface/lerobot/calibration/virtual_robots/bi_so100"
            else:
                output_dir = Path(output_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create comprehensive calibration file
            filename = f"{self.leader_id}_to_{self.virtual_arm}_arm_interactive.json"
            filepath = output_dir / filename
            
            # Prepare data for saving
            save_data = {
                "metadata": {
                    "generated_by": "InteractiveBiSO100Calibrator",
                    "leader_id": self.leader_id,
                    "arm_mapping": f"{self.leader_id} → {self.virtual_arm} virtual arm",
                    "virtual_arm": self.virtual_arm,
                    "environment_id": self.env_id,
                    "calibration_points": len(self.calibration_points),
                    "method": "interactive_manual_positioning",
                    "joint_mapping": {
                        leader_joint: data["virtual_joint"] 
                        for leader_joint, data in calibration_data.items()
                    }
                },
                "calibration_data": calibration_data
            }
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            print(f"\n✅ Calibration saved to: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving calibration: {e}")
            return False

    def run_interactive_calibration(self):
        """Main interactive calibration loop"""
        pygame.init()
        
        screen_width, screen_height = 800, 900
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Interactive BiSO100 Calibration")
        font = pygame.font.SysFont(None, 24)
        small_font = pygame.font.SysFont(None, 18)
        
        print("\n" + "="*60)
        print("INTERACTIVE CALIBRATION MODE")
        print("="*60)
        print("VIRTUAL ARM CONTROLS:")
        print("  Q/A: Shoulder Pan +/-    W/S: Shoulder Lift +/-")
        print("  E/D: Elbow Flex +/-      R/F: Wrist Flex +/-")
        print("  T/G: Wrist Roll +/-      Z: Toggle Gripper")
        print("  ESC: Reset to default pose")
        print("")
        print("CALIBRATION CONTROLS:")
        print("  SPACE: Record position pair")
        print("  C: Calculate & save calibration")
        print("  ~ (tilde): Quit")
        print("")
        print("WORKFLOW:")
        print("1. Position both arms to the SAME pose")
        print("2. Press SPACE to record the position")
        print("3. Repeat for multiple poses (minimum 2)")
        print("4. Press C to calculate and save calibration")
        print("="*60)
        
        action = np.zeros(12)
        step_counter = 0
        warmup_steps = 50
        show_help = False  # Help display toggle
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_BACKQUOTE:  # Backtick/tilde key (~)
                        print("Quitting...")
                        return False
                    elif event.key == pygame.K_SPACE:
                        # Record calibration point
                        description = f"Point_{len(self.calibration_points)+1}"
                        self.record_calibration_point(description)
                    elif event.key == pygame.K_c:
                        # Calculate and save calibration
                        if len(self.calibration_points) >= 2:
                            try:
                                calibration_data = self.calculate_calibration()
                                success = self.save_calibration(calibration_data)
                                if success:
                                    print("\n✅ Calibration complete! You can now use the calibration for teleoperation.")
                                    return True
                                else:
                                    print("\n❌ Failed to save calibration")
                            except Exception as e:
                                print(f"\n❌ Calibration calculation failed: {e}")
                        else:
                            print(f"\n⚠️  Need at least 2 calibration points (currently have {len(self.calibration_points)})")
                    elif event.key == pygame.K_ESCAPE:
                        # Reset target joints
                        self.target_joints = np.zeros_like(self.target_joints)
                        if self.virtual_arm == "left":
                            self.target_joints[1] = 0.303  # Default pose
                            self.target_joints[2] = 0.556
                        else:  # right
                            self.target_joints[7] = 0.303
                            self.target_joints[8] = 0.556
                    elif event.key == pygame.K_h:
                        # Toggle help display
                        show_help = not show_help
            
            # Virtual arm keyboard control (from demo_bi_so100_ctrl.py)
            keys = pygame.key.get_pressed()
            
            if step_counter >= warmup_steps:
                # Determine which arm to control based on virtual_arm setting
                if self.virtual_arm == "left":
                    arm_offset = 0  # Control joints 0-5
                else:  # right
                    arm_offset = 6  # Control joints 6-11
                
                # Arm control
                if keys[pygame.K_q]: self.target_joints[arm_offset + 0] += self.joint_step
                if keys[pygame.K_a]: self.target_joints[arm_offset + 0] -= self.joint_step
                if keys[pygame.K_w]: self.target_joints[arm_offset + 1] += self.joint_step
                if keys[pygame.K_s]: self.target_joints[arm_offset + 1] -= self.joint_step
                if keys[pygame.K_e]: self.target_joints[arm_offset + 2] += self.joint_step
                if keys[pygame.K_d]: self.target_joints[arm_offset + 2] -= self.joint_step
                if keys[pygame.K_r]: self.target_joints[arm_offset + 3] += self.joint_step
                if keys[pygame.K_f]: self.target_joints[arm_offset + 3] -= self.joint_step
                if keys[pygame.K_t]: self.target_joints[arm_offset + 4] += self.joint_step
                if keys[pygame.K_g]: self.target_joints[arm_offset + 4] -= self.joint_step
                
                # Gripper control
                if keys[pygame.K_z]:
                    if self.target_joints[arm_offset + 5] < 0.4:
                        self.target_joints[arm_offset + 5] = 1.5
                    else:
                        self.target_joints[arm_offset + 5] = 0.1
                    pygame.time.delay(200)
            
            # Apply control
            current_joints = self._get_mapped_joints(self.robot)
            
            if step_counter < warmup_steps:
                action = np.zeros_like(action)
            else:
                for i in range(len(action)):
                    action[i] = self.p_gain[i] * (self.target_joints[i] - current_joints[i])
            
            # Update environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.env.render()
            
            # Display interface
            screen.fill((0, 0, 0))
            
            y_pos = 10
            
            # Title
            title_text = font.render("Interactive BiSO100 Calibration", True, (255, 255, 255))
            screen.blit(title_text, (10, y_pos))
            y_pos += 40
            
            # Warmup status
            if step_counter < warmup_steps:
                warmup_text = font.render(f"WARMUP: {step_counter}/{warmup_steps} steps", True, (255, 0, 0))
                screen.blit(warmup_text, (10, y_pos))
                y_pos += 30
            
            # Instructions
            instructions = [
                "=== VIRTUAL ARM CONTROL ===",
                "Q/A: Shoulder Pan (Rotation) +/-",
                "W/S: Shoulder Lift (Pitch) +/-", 
                "E/D: Elbow Flex +/-",
                "R/F: Wrist Flex (Pitch) +/-",
                "T/G: Wrist Roll +/-",
                "Z: Toggle Gripper (Open/Close)",
                "ESC: Reset arm to default pose",
                "",
                "=== CALIBRATION CONTROL ===",
                "SPACE: Record current position pair",
                "C: Calculate & save calibration (≥2 pts)",
                "~ (tilde): Quit program",
                "H: Toggle help overlay",
                "",
                "=== WORKFLOW ===",
                "1. Move BOTH arms to SAME pose",
                "2. Press SPACE to record",
                "3. Repeat 4-6 different poses", 
                "4. Press C when done"
            ]
            
            for instruction in instructions:
                inst_text = small_font.render(instruction, True, (200, 200, 200))
                screen.blit(inst_text, (10, y_pos))
                y_pos += 20
            
            y_pos += 20
            
            # Current positions
            try:
                leader_pos = self.get_leader_positions()
                virtual_pos = self.get_virtual_positions()
                
                # Leader positions
                leader_text = font.render("Leader Arm (degrees):", True, (255, 255, 0))
                screen.blit(leader_text, (10, y_pos))
                y_pos += 25
                
                for joint, pos in leader_pos.items():
                    pos_text = small_font.render(f"  {joint}: {pos:.1f}°", True, (255, 255, 200))
                    screen.blit(pos_text, (10, y_pos))
                    y_pos += 18
                
                y_pos += 10
                
                # Virtual positions
                virtual_text = font.render(f"Virtual {self.virtual_arm.capitalize()} Arm (radians):", True, (0, 255, 0))
                screen.blit(virtual_text, (10, y_pos))
                y_pos += 25
                
                for joint, pos in virtual_pos.items():
                    pos_text = small_font.render(f"  {joint}: {pos:.3f} rad ({np.degrees(pos):.1f}°)", True, (200, 255, 200))
                    screen.blit(pos_text, (10, y_pos))
                    y_pos += 18
                
                y_pos += 20
                
                # Calibration points
                points_text = font.render(f"Recorded Points: {len(self.calibration_points)}", True, (0, 255, 255))
                screen.blit(points_text, (10, y_pos))
                y_pos += 30
                
                for i, point in enumerate(self.calibration_points[-5:]):  # Show last 5 points
                    point_text = small_font.render(f"Point {i+1}: {point.description}", True, (150, 255, 255))
                    screen.blit(point_text, (20, y_pos))
                    y_pos += 18
                
            except Exception as e:
                error_text = font.render(f"Error: {e}", True, (255, 0, 0))
                screen.blit(error_text, (10, y_pos))
            
            # Help overlay
            if show_help:
                overlay = pygame.Surface((screen_width, screen_height))
                overlay.set_alpha(220)
                overlay.fill((0, 0, 0))
                screen.blit(overlay, (0, 0))
                
                # Help content
                help_title = font.render("KEYBOARD REFERENCE", True, (255, 255, 255))
                screen.blit(help_title, (50, 50))
                
                help_content = [
                    "VIRTUAL ARM CONTROL (Hold keys):",
                    "  Q/A: Shoulder Pan ±     W/S: Shoulder Lift ±",
                    "  E/D: Elbow Flex ±       R/F: Wrist Flex ±", 
                    "  T/G: Wrist Roll ±       Z: Toggle Gripper",
                    "  ESC: Reset to default",
                    "",
                    "CALIBRATION (Single press):",
                    "  SPACE: Record position pair",
                    "  C: Calculate & save calibration", 
                    "  ~: Quit program",
                    "",
                    "WORKFLOW:",
                    "1. Move both arms to SAME pose",
                    "2. Press SPACE to record",
                    "3. Repeat for 4-6 poses",
                    "4. Press C when done",
                    "",
                    "Release H to continue..."
                ]
                
                for i, line in enumerate(help_content):
                    color = (255, 255, 255) if not line.startswith("  ") else (200, 200, 200)
                    if line.startswith("WORKFLOW:") or line.startswith("VIRTUAL") or line.startswith("CALIBRATION"):
                        color = (255, 255, 0)
                    help_line = small_font.render(line, True, color)
                    screen.blit(help_line, (50, 90 + i * 20))
                
            pygame.display.flip()
            step_counter += 1
            time.sleep(0.01)
        
        return False

    def cleanup(self):
        """Clean up resources"""
        if self.teleop:
            self.teleop.disconnect()
        if self.env:
            self.env.close()
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive BiSO100 calibration using manual positioning"
    )
    parser.add_argument(
        "--leader-id", 
        type=str, 
        default="right_leader",
        help="Leader arm identifier"
    )
    parser.add_argument(
        "--virtual-arm", 
        type=str, 
        choices=["left", "right"], 
        default="right",
        help="Which virtual arm to calibrate"
    )
    parser.add_argument(
        "--teleop-port", 
        type=str, 
        default="/dev/ttyACM0",
        help="SO100 teleoperator port"
    )
    parser.add_argument(
        "--env-id", 
        type=str, 
        default="BiSO100OpenLid-v1",
        help="ManiSkill environment ID"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        help="Output directory (default: ~/.cache/huggingface/lerobot/calibration/virtual_robots/bi_so100/)"
    )
    
    args = parser.parse_args()
    
    print("Interactive BiSO100 Calibration Generator")
    print("="*50)
    print(f"Leader ID: {args.leader_id}")
    print(f"Virtual arm: {args.virtual_arm}")
    print(f"Teleop port: {args.teleop_port}")
    print(f"Environment: {args.env_id}")
    print("="*50)
    
    calibrator = InteractiveBiSO100Calibrator(
        leader_id=args.leader_id,
        virtual_arm=args.virtual_arm,
        teleop_port=args.teleop_port,
        env_id=args.env_id
    )
    
    try:
        # Setup
        calibrator.setup_environment()
        calibrator.setup_teleoperator()
        
        # Run interactive calibration
        success = calibrator.run_interactive_calibration()
        
        if success:
            print("\n🎉 Interactive calibration completed successfully!")
            print("You can now use the generated calibration for teleoperation.")
        else:
            print("\n❌ Calibration was cancelled or failed.")
            return 1
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    finally:
        calibrator.cleanup()
    
    return 0


if __name__ == "__main__":
    exit(main())