# AMP extension based on SKRL
It is designed for Isaaclab Manager Based Env base on SKRL. You donnot need to install any extra extension in your env.
## Train Your Own Robot
1. In your isaaclab manager, please manually set your joint orders to match the order of motion dataset. e.g.:
In ActionsCfg, you should set joint pos as below:
```
joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[
                                                                            "left_hip_pitch_joint",
                                                                            "left_hip_roll_joint",
                                                                            "left_hip_yaw_joint",
                                                                            "left_knee_joint",
                                                                            "left_ankle_pitch_joint",
                                                                            "left_ankle_roll_joint",
                                                                            "right_hip_pitch_joint",
                                                                            "right_hip_roll_joint",
                                                                            "right_hip_yaw_joint",
                                                                            "right_knee_joint",
                                                                            "right_ankle_pitch_joint",
                                                                            "right_ankle_roll_joint",
                                                                            "waist_yaw_joint",
                                                                            # "waist_roll_joint",
                                                                            # "waist_pitch_joint",
                                                                            "left_shoulder_pitch_joint",
                                                                            "left_shoulder_roll_joint",
                                                                            "left_shoulder_yaw_joint",
                                                                            "left_elbow_joint",
                                                                            "left_wrist_roll_joint",
                                                                            # "left_wrist_pitch_joint",
                                                                            # "left_wrist_yaw_joint",
                                                                            "right_shoulder_pitch_joint",
                                                                            "right_shoulder_roll_joint",
                                                                            "right_shoulder_yaw_joint",
                                                                            "right_elbow_joint",
                                                                            "right_wrist_roll_joint",
                                                                            # "right_wrist_pitch_joint",
                                                                            # "right_wrist_yaw_joint"
                                                                        ], 
                                                                            scale=0.5, use_default_offset=True, preserve_order=True)
```
In ObservationsCfg, you should set joint pos and joint vel below:
```
joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), history_length=5, 
                                                                    params={"asset_cfg": SceneEntityCfg("robot", 
                                                                    joint_names=[
                                                                            "left_hip_pitch_joint",
                                                                            "left_hip_roll_joint",
                                                                            "left_hip_yaw_joint",
                                                                            "left_knee_joint",
                                                                            "left_ankle_pitch_joint",
                                                                            "left_ankle_roll_joint",
                                                                            "right_hip_pitch_joint",
                                                                            "right_hip_roll_joint",
                                                                            "right_hip_yaw_joint",
                                                                            "right_knee_joint",
                                                                            "right_ankle_pitch_joint",
                                                                            "right_ankle_roll_joint",
                                                                            "waist_yaw_joint",
                                                                            "left_shoulder_pitch_joint",
                                                                            "left_shoulder_roll_joint",
                                                                            "left_shoulder_yaw_joint",
                                                                            "left_elbow_joint",
                                                                            "left_wrist_roll_joint",
                                                                            "right_shoulder_pitch_joint",
                                                                            "right_shoulder_roll_joint",
                                                                            "right_shoulder_yaw_joint",
                                                                            "right_elbow_joint",
                                                                            "right_wrist_roll_joint",
                                                                        ], preserve_order=True)} )
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5), history_length=5, 
                                                                    params={"asset_cfg": SceneEntityCfg("robot", 
                                                                    joint_names=[
                                                                            "left_hip_pitch_joint",
                                                                            "left_hip_roll_joint",
                                                                            "left_hip_yaw_joint",
                                                                            "left_knee_joint",
                                                                            "left_ankle_pitch_joint",
                                                                            "left_ankle_roll_joint",
                                                                            "right_hip_pitch_joint",
                                                                            "right_hip_roll_joint",
                                                                            "right_hip_yaw_joint",
                                                                            "right_knee_joint",
                                                                            "right_ankle_pitch_joint",
                                                                            "right_ankle_roll_joint",
                                                                            "waist_yaw_joint",
                                                                            # "waist_roll_joint",
                                                                            # "waist_pitch_joint",
                                                                            "left_shoulder_pitch_joint",
                                                                            "left_shoulder_roll_joint",
                                                                            "left_shoulder_yaw_joint",
                                                                            "left_elbow_joint",
                                                                            "left_wrist_roll_joint",
                                                                            # "left_wrist_pitch_joint",
                                                                            # "left_wrist_yaw_joint",
                                                                            "right_shoulder_pitch_joint",
                                                                            "right_shoulder_roll_joint",
                                                                            "right_shoulder_yaw_joint",
                                                                            "right_elbow_joint",
                                                                            "right_wrist_roll_joint"
                                                                            # "right_wrist_pitch_joint",
                                                                            # "right_wrist_yaw_joint"
                                                                        ], preserve_order=True)})
```

2. in `/utils/amp_states_collection.py`:
please review the observation manager of your environment first, e.g.:
```
[INFO] Observation Manager: <ObservationManager> contains 1 groups.
            +---------------------------------------------------------+
            | Active Observation Terms in Group: 'policy' (shape: (81,)) |
            +-----------+---------------------------------+-----------+
            |   Index   | Name                            |   Shape   |
            +-----------+---------------------------------+-----------+
            |     0     | base_lin_vel                    |    (3,)   |
            |     1     | base_ang_vel                    |    (3,)   |
            |     2     | projected_gravity               |    (3,)   |
            |     3     | velocity_commands               |    (3,)   |
            |     5     | joint_pos                       |   (23,)   |
            |     6     | joint_vel                       |   (23,)   |
            |     7     | actions                         |   (23,)   |
            +-----------+---------------------------------+-----------+
```
If your dataset need joint_pos and joint velocity, then you should concatenate joint pos and joint vel to match the format of your motion dataset
3. In `/utils/load_motion_dataset.py`:
Assume that the motion dataset contains keys of "states" and "next_states".
You should review the default joint offset in your robot_cfg.py, e.g.:
```
G1_CFG = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(CURRENT_DIR, "usd/g1_23dof_rev_1_0.usd"),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=3.0,
                max_angular_velocity=3.0,
                max_depenetration_velocity=10.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.8),
            joint_pos={
                ".*_hip_pitch_joint": -0.20,
                ".*_knee_joint": 0.42,
                ".*_ankle_pitch_joint": -0.23,
                # ".*_elbow_pitch_joint": 0.87,
                "left_shoulder_roll_joint": 0.16,
                "left_shoulder_pitch_joint": 0.35,
                "right_shoulder_roll_joint": -0.16,
                "right_shoulder_pitch_joint": 0.35,
            },
```
Then you also set the offset of your motion dataset. Then concatenate states and next_states to make the referenced amp_states match the format in step 2

## Prepare Your Motion Dataset
There's a demo of transform motion dataset of GMR to the format of AMP training in `/convert_dataset/convert_dataset.py`. Only dof_pos can successfully train AMP locomotion.

## Start Training
in your conda env, run:
```
python train_amp.py --num_envs 4096 --headless
```

## Possible Issues of Training AMP
You can see AMP as an extra reward term of traditional RL training. For generalization, it is recommended to set large task reward weight and small style reward weight( e.g. set task reward weight to 0.9 and set style reward weight to 0.1)

AMP training highly relies on the quality of motion dataset. If you fail to train AMP with this code. One common issue is the feet part.

### Solution 1 
Please remove the feet part from AMP dataset and restart training.


#### Original Dataset
<video src="https://github.com/user-attachments/assets/c52d825b-c9bf-47a7-b3ce-be6731de52b9" width="400" controls></video>


#### Dataset Without Feet
<video src="https://github.com/user-attachments/assets/650623f0-cc19-49ba-9342-72d198bc71c4" width="400" controls></video>





### Solution 2
Please manually refine your motion dataset to make it reliable.
