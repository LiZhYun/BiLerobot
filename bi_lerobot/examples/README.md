# BiLerobot Examples

This directory contains example scripts for working with the BiSO100 dual-arm robot in ManiSkill environments using the LeRobot framework. These examples demonstrate various aspects of robot control, data collection, and teleoperation.

## Prerequisites

Before running any examples, make sure you have the required packages installed:

```bash
pip install maniskill
pip install lerobot
```

Additionally, you'll need to install this package:

```bash
pip install -e .
```

## Available Examples

### 1. `record_bi_so100_maniskill.py` - Dataset Recording

Records robot demonstrations in ManiSkill environments using the LeRobot dataset format. Supports multiple operation modes:

#### Teleoperator Mode (using real SO100 leader arms)
Control the virtual robot using physical SO100 leader arms for teleoperation:

```bash
python bi_lerobot/examples/record_bi_so100_maniskill.py \
    --robot.env_id=BiSO100OpenLid-v1 \
    --robot.leader_ids=left_leader,right_leader \
    --robot.teleop_ports=/dev/ttyACM0,/dev/ttyACM1 \
    --robot.calibration_files="path/to/left_calibration.json,path/to/right_calibration.json" \
    --dataset.repo_id=username/bi_so100_maniskill_teleop \
    --dataset.num_episodes=10 \
    --dataset.single_task="Open the bottle lid with both arms"
```

#### Policy Mode (using a pretrained policy)
Generate demonstrations using a pretrained policy:

```bash
python bi_lerobot/examples/record_bi_so100_maniskill.py \
    --robot.env_id=BiSO100OpenLid-v1 \
    --policy.path=path/to/pretrained/policy \
    --dataset.repo_id=username/bi_so100_maniskill_policy \
    --dataset.num_episodes=10 \
    --dataset.single_task="Open the bottle lid with both arms" \
    --display_data=true
```

#### Manual Mode (for testing)
Test the environment without teleoperators or policy:

```bash
python bi_lerobot/examples/record_bi_so100_maniskill.py \
    --robot.env_id=BiSO100OpenLid-v1 \
    --dataset.repo_id=username/bi_so100_maniskill_manual \
    --dataset.num_episodes=1 \
    --dataset.single_task="Test environment"
```

**Key Features:**
- LeRobot dataset format compatibility
- Multiple control modes (teleop, policy, manual)
- Configurable recording parameters (FPS, episode duration, etc.)
- Automatic data upload to Hugging Face Hub

### 2. `teleoperate_bi_so100_with_real_leader.py` - Real-time Teleoperation

Direct teleoperation of the virtual BiSO100 robot using physical SO100 leader arms:

```bash
python bi_lerobot/examples/teleoperate_bi_so100_with_real_leader.py \
    --leader-ids=left_leader,right_leader \
    --teleop-ports=/dev/ttyACM0,/dev/ttyACM1 \
    --calibration-files=path/to/left.json,path/to/right.json \
    --env-id=BiSO100OpenLid-v1 \
    --fps=30
```

**Key Features:**
- Real-time control with customizable FPS
- Automatic calibration loading and arm mapping
- Support for both single and dual-arm teleoperation
- Leader-to-virtual arm position mapping

### 3. `generate_bi_so100_calibration_interactive.py` - Interactive Calibration

Generate calibration data by manually positioning both physical leader and virtual arms:

```bash
python bi_lerobot/examples/generate_bi_so100_calibration_interactive.py \
    --leader-id=right_leader \
    --virtual-arm=right \
    --teleop-port=/dev/ttyACM1
```

**Interactive Controls:**
- **Physical leader arm:** Move manually to desired positions
- **Virtual arm:** Use keyboard controls (WASD for movement, QE for rotation, etc.)
- **SPACE:** Record current position correspondence
- **C:** Calculate and save calibration
- **Q:** Quit

**Key Features:**
- Manual position correspondence recording
- Real-time calibration calculation
- Automatic calibration file generation
- Support for both left and right arms

### 4. `demo_bi_so100_ctrl.py` - Basic Joint Control Demo

Demonstrates basic keyboard-controlled joint-level control of the BiSO100 robot:

```bash
python bi_lerobot/examples/demo_bi_so100_ctrl.py \
    --env-id=BiSO100OpenLid-v1 \
    --render-mode=human
```

**Keyboard Controls:**
- Various keys for controlling individual joints of both arms
- Real-time visualization of robot movement
- Adjustable joint step size and control gains

**Key Features:**
- Direct joint control interface
- Real-time robot visualization
- Customizable control parameters
- Educational tool for understanding robot kinematics

### 5. `demo_bi_so100_ctrl_ee.py` - End-Effector Control Demo

Demonstrates end-effector (Cartesian) control of the BiSO100 robot with inverse kinematics:

```bash
python bi_lerobot/examples/demo_bi_so100_ctrl_ee.py \
    --env-id=BiSO100OpenLid-v1 \
    --render-mode=human
```

**Key Features:**
- End-effector position control
- Inverse kinematics calculations
- 2-link arm modeling
- Workspace boundary handling
- Keyboard-based Cartesian control

## Common Parameters

### Environment IDs
- `BiSO100OpenLid-v1` - Bottle lid opening task
- Other BiSO100 environments (check ManiSkill documentation)

### Render Modes
- `human` - Visual rendering with GUI
- `none` - No rendering (faster execution)

### Control Modes
- `pd_joint_delta_pos_dual_arm` - Position control for dual arms
- Other control modes supported by ManiSkill

## Hardware Requirements

### For Teleoperation Examples
- Physical SO100 leader arm(s)
- USB serial connections (typically `/dev/ttyACM0`, `/dev/ttyACM1`)
- Proper calibration files

### For Demo Examples
- Standard computer with graphics support
- Keyboard for control input

## Calibration Workflow

1. **Generate Calibration:** Use `generate_bi_so100_calibration_interactive.py` to create calibration files
2. **Test Teleoperation:** Use `teleoperate_bi_so100_with_real_leader.py` to verify calibration
3. **Record Data:** Use `record_bi_so100_maniskill.py` with teleop mode for data collection

## Troubleshooting

### Common Issues

1. **Serial Port Access:**
   ```bash
   sudo chmod 666 /dev/ttyACM*
   ```

2. **Missing Calibration Files:**
   - Run the calibration script first
   - Check file paths in commands
   - Ensure calibration files exist in specified locations

3. **Environment Loading Errors:**
   - Verify ManiSkill installation
   - Check if BiLerobot package is properly installed
   - Ensure all dependencies are available

4. **Control Responsiveness:**
   - Adjust FPS parameter
   - Check system performance
   - Verify hardware connections

## File Structure

```
bi_lerobot/examples/
├── README.md                                    # This file
├── record_bi_so100_maniskill.py                # Data recording
├── teleoperate_bi_so100_with_real_leader.py    # Real-time teleoperation
├── generate_bi_so100_calibration_interactive.py # Calibration generation
├── demo_bi_so100_ctrl.py                       # Basic joint control
└── demo_bi_so100_ctrl_ee.py                    # End-effector control
```

## Additional Resources

- [LeRobot Documentation](https://huggingface.co/docs/lerobot)
- [ManiSkill Documentation](https://maniskill.readthedocs.io/)
- [BiSO100 Robot Specifications](https://www.bilibili.com/read/cv34492412/)

## Contributing

When adding new examples:
1. Follow the existing code structure and naming conventions
2. Add comprehensive docstrings and comments
3. Update this README with new example descriptions
4. Test examples with different configurations

For questions or issues, please refer to the main repository documentation or create an issue on the project repository. 