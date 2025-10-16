import torch
import numpy as np
import os 

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_motion_dataset(amp_observation_space):
    """Load and prepare motion dataset"""
    try:
        file_path=os.path.join(CURRENT_DIR, "state_next_state_dataset.npz")
        motion_data = np.load(file_path)
          
        # Verify required keys
        assert 'states' in motion_data and 'next_states' in motion_data, \
            "Dataset must contain 'state' and 'next_states' keys"
          
        # Combine state-next_state pairs
        states = motion_data['states']
        next_states = motion_data['next_states']
          
        # Check dimensions
        assert states.shape[0] == next_states.shape[0], \
            f"Mismatched samples: {states.shape[0]} vs {next_states.shape[0]}"
        assert states.shape[1] + next_states.shape[1] == amp_observation_space, \
            f"Invalid feature dimension: {states.shape[1]}+{next_states.shape[1]}"
            
        offset_vec = np.zeros(19, dtype=np.float32)
            
        patterns = {
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.42,
            # ".*_ankle_pitch_joint": -0.23,
            "left_shoulder_roll_joint": 0.16,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.16,
            "right_shoulder_pitch_joint": -0.35,
        }
        
        # joint_names = [
        #         "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        #         "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        #         "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        #         "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        #         "waist_yaw_joint", "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
        #         "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
        #         "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
        #         "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
        #     ]
        joint_names = [
                "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
                "left_knee_joint", 
                "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
                "right_knee_joint", 
                "waist_yaw_joint", "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
                "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
            ]

        # Create mapping from patterns to offsets
        for i, name in enumerate(joint_names):
            # Hip pitch joints
            if name in ["left_hip_pitch_joint", "right_hip_pitch_joint"]:
                offset_vec[i] = patterns[".*_hip_pitch_joint"]
            # Knee joints
            elif name in ["left_knee_joint", "right_knee_joint"]:
                offset_vec[i] = patterns[".*_knee_joint"]
            # Ankle pitch joints
            elif name in ["left_ankle_pitch_joint", "right_ankle_pitch_joint"]:
                offset_vec[i] = patterns[".*_ankle_pitch_joint"]
            # Other joints
            elif name in patterns:
                offset_vec[i] = patterns[name]
            # Default to 0 offset (no match)
        print("offset to motion data:")
        print(offset_vec)

        # Add offsets to joint positions
        states[:, :19] -= offset_vec
        next_states[:, :19] -= offset_vec

            # states[:,[9, 10, 11, 12, 14, 15, 16, 28, 29, 30, 31, 33, 34, 35]] *= -1
            # next_states[:, [9, 10, 11, 12, 14, 15, 16, 28, 29, 30, 31, 33, 34, 35]] *= -1

            # left_elbow
            # left shoulder pitch
            # right shoulder pitch
            # left shoulder roll
            # right shoulder roll
            # left shoulder yaw
            # right shoulder yaw

          
            # Concatenate features
        combined = np.concatenate([states, next_states], axis=1)
          
        # Convert to tensor and move to device
        motion_tensor = torch.from_numpy(combined).float()
          
        print(f"Loaded motion dataset: {motion_tensor.shape}")
        return motion_tensor
          
    except Exception as e:
        raise RuntimeError(f"Failed loading dataset: {str(e)}")

# load_motion_dataset(46)