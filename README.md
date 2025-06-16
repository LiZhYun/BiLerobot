# BiLerobot: Bimanual SO100 Digital Twin

A bimanual robotics platform combining **LeRobot** and **ManiSkill** for advanced dual-arm manipulation tasks using the SO100 robot digital twin.

## 🚀 Overview

BiLerobot is a comprehensive robotics simulation and learning framework that provides:

- **Bimanual SO100 Robot**: Digital twin of the SO100 dual-arm manipulation platform
- **ManiSkill Integration**: High-fidelity physics simulation environments
- **LeRobot Compatibility**: Dataset recording, policy training, and deployment
- **Teleoperation Support**: Real-time control with physical SO100 leader arms
- **Multi-Camera System**: Wrist-mounted cameras for visual feedback

## 📋 Requirements

### Core Dependencies
- **Python**: ≥3.8
- **ManiSkill**: Physics simulation and environment framework
- **LeRobot**: Dataset management and policy training
- **PyTorch**: Deep learning backend
- **SAPIEN**: 3D physics simulation engine
- **Gymnasium**: Reinforcement learning environment interface

### Hardware (Optional)
- SO100 Leader Arms for teleoperation
- USB connections for real-time control

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/BiLerobot.git
cd BiLerobot
```

2. **Install the package:**
```bash
pip install -e .
```

3. **Install dependencies:**
```bash
# Install ManiSkill
pip install mani-skill

# Install LeRobot
pip install lerobot

# Additional dependencies
pip install sapien gymnasium torch transforms3d pygame tyro
```

## 🤖 Robot Configuration

### BiSO100 Specifications
- **Dual Arms**: 2x SO100 robotic arms
- **DOF**: 5 joints per arm + 1 gripper = 12 total DOF
- **Mounting**: Fixed base (stationary platform)
- **Cameras**: Wrist-mounted cameras on each arm
- **Control Modes**: Position, velocity, delta position, end-effector control

### Joint Configuration
```
Left Arm:  [Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw]
Right Arm: [Rotation_2, Pitch_2, Elbow_2, Wrist_Pitch_2, Wrist_Roll_2, Jaw_2]
```

## 🌍 Environments

### Available Tasks

#### BiSO100OpenLid-v1
- **Objective**: Collaborative lid opening using both arms
- **Features**: Bottle with removable lid, coordinated manipulation
- **Cameras**: Top-down and side view for complete workspace coverage
- **Success**: Lid successfully removed from bottle

### Environment Features
- **Physics**: SAPIEN-based realistic simulation
- **Rendering**: Ray-traced visuals with configurable shaders
- **Sensors**: Multi-camera RGB/depth observation
- **Rewards**: Dense and sparse reward modes

## 🎮 Usage Examples

### 1. Basic Robot Control
```bash
# Run interactive control demo
python bi_lerobot/examples/demo_bi_so100_ctrl.py \
    --env_id BiSO100OpenLid-v1 \
    --render_mode human \
    --control_mode pd_joint_delta_pos_dual_arm
```

### 2. End-Effector Control
```bash
# Control robot end-effectors directly
python bi_lerobot/examples/demo_bi_so100_ctrl_ee.py \
    --env_id BiSO100OpenLid-v1 \
    --control_mode pd_ee_delta_pose
```

### 3. Teleoperation with Real Hardware
```bash
# Control simulation with physical SO100 leaders
python bi_lerobot/examples/teleoperate_bi_so100_with_real_leader.py \
    --env_id BiSO100OpenLid-v1 \
    --leader_ids left_leader,right_leader \
    --teleop_ports /dev/ttyACM0,/dev/ttyACM1
```

### 4. Dataset Recording
```bash
# Record demonstration data with teleoperation
python bi_lerobot/examples/record_bi_so100_maniskill.py \
    --robot.env_id BiSO100OpenLid-v1 \
    --robot.leader_ids left_leader,right_leader \
    --robot.teleop_ports /dev/ttyACM0,/dev/ttyACM1 \
    --dataset.repo_id username/bi_so100_demonstrations \
    --dataset.num_episodes 50 \
    --dataset.single_task "Open bottle lid with both arms"
```

### 5. Policy Training Integration
```bash
# Record data using a pretrained policy
python bi_lerobot/examples/record_bi_so100_maniskill.py \
    --robot.env_id BiSO100OpenLid-v1 \
    --policy.path path/to/pretrained/policy \
    --dataset.repo_id username/bi_so100_policy_data \
    --dataset.num_episodes 100
```

## 🎯 Control Modes

### Supported Control Modes
- `pd_joint_pos`: Direct joint position control
- `pd_joint_delta_pos_dual_arm`: Incremental joint position (dual-arm)
- `pd_ee_delta_pos`: End-effector position control
- `pd_ee_delta_pose`: End-effector pose control (position + orientation)
- `pd_joint_vel`: Joint velocity control

### Key Mappings (Interactive Control)
- **WASD**: Move end-effector in XY plane
- **QE**: Move end-effector up/down
- **RF**: Rotate wrist pitch
- **TG**: Rotate wrist roll
- **Space**: Toggle gripper open/close
- **Arrow Keys**: Control second arm
- **Enter**: Reset environment

## 📊 Data Collection

### Dataset Format
- **LeRobot Compatible**: Standard HuggingFace dataset format
- **Multi-Modal**: RGB cameras, joint states, actions
- **Timestamped**: Synchronized data streams
- **Compressed**: Efficient video encoding for large datasets

### Recording Features
- Real-time teleoperation recording
- Policy rollout recording
- Multi-camera synchronized capture
- Automatic episode segmentation
- Data validation and quality checks

## 🔧 Calibration

### SO100 Leader Calibration
```bash
# Generate interactive calibration for teleoperation
python bi_lerobot/examples/generate_bi_so100_calibration_interactive.py \
    --robot_id bi_so100 \
    --leader_ids left_leader,right_leader \
    --output_dir ~/.cache/huggingface/lerobot/calibration/virtual_robots/bi_so100/
```

## 📁 Project Structure

```
bi_lerobot/
├── agents/           # Robot agent implementations
│   └── robots/
│       └── bi_so100/ # SO100 robot definition and control
├── envs/            # Simulation environments
│   └── tasks/
│       └── tabletop/ # Tabletop manipulation tasks
├── assets/          # Robot models and configurations
│   └── robots/
│       └── bi_so100/ # URDF and mesh files
└── examples/        # Usage examples and demos
    ├── demo_bi_so100_ctrl.py              # Basic robot control
    ├── demo_bi_so100_ctrl_ee.py           # End-effector control
    ├── teleoperate_bi_so100_with_real_leader.py # Hardware teleoperation
    ├── record_bi_so100_maniskill.py       # Dataset recording
    └── generate_bi_so100_calibration_interactive.py # Calibration
```

## 🧪 Development

### Adding New Tasks
1. Create task class in `bi_lerobot/envs/tasks/`
2. Register environment with `@register_env` decorator
3. Implement required methods: `_load_scene`, `_initialize_episode`, `evaluate`

### Extending Robot Capabilities
1. Modify robot URDF in `bi_lerobot/assets/robots/bi_so100/`
2. Update control configurations in `bi_so100.py`
3. Add new control modes as needed

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project follows the respective licenses of its components:
- **LeRobot**: Apache 2.0 License
- **ManiSkill**: MIT License

## 🙏 Acknowledgments

- **LeRobot Team**: Dataset management and policy framework
- **ManiSkill Team**: Physics simulation environment
- **SAPIEN**: 3D physics simulation engine
- **HuggingFace**: Dataset hosting and management

## 📞 Support

For issues and questions:
1. Check the [Issues](https://github.com/your-username/BiLerobot/issues) page
2. Review ManiSkill and LeRobot documentation
3. Join the community discussions

---

**Built with ❤️ for advancing bimanual robotics research** 