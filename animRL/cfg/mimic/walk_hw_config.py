from animRL.cfg.mimic.mimic_pi_config import MimicCfg, MimicTrainCfg
RANDOMIZE = True  

class WalkHWCfg(MimicCfg):
    class env(MimicCfg.env):
        num_envs = 4096

        num_actions = 12
        num_observations = 43
        obs_history_len = 5
        episode_length = 150  # episode length

        reference_state_initialization = True  # initialize state from reference data

    class motion_loader(MimicCfg.motion_loader):
        motion_files = '{ROOT_DIR}/resources/datasets/pi/Walk.txt'

    class rewards(MimicCfg.rewards):
        class terms:

            joint_targets_rate = [5.0, 0.0]

            track_base_height = [0.1, 0.0]
            track_base_orientation = [0.5, 0.0]
            track_joint_pos = [1.8, 0.0]
            track_base_vel = [1.0, 0.0]
            track_ee_pos = [0.25, 0.0]

    class control(MimicCfg.control):
        control_type = 'P'  # P: position, V: velocity, T: torques
        stiffness = {
            "hip_pitch_joint": 50.0,
            "hip_roll_joint": 25.0,
            "thigh_joint": 25.0,
            "calf_joint": 50.0,
            "ankle_pitch_joint": 50.0,
            "ankle_roll_joint": 25.0,
        }
        damping = {
            "hip_pitch_joint": 0.8,
            "hip_roll_joint": 0.5,
            "thigh_joint": 0.5,
            "calf_joint": 0.8,
            "ankle_pitch_joint": 0.8,
            "ankle_roll_joint": 0.5,
        }

    class domain_rand(MimicCfg.domain_rand):
        randomize_friction = RANDOMIZE
        friction_range = [0.6, 1.2] if RANDOMIZE else [0.9, 1.0]
        randomize_base_mass = RANDOMIZE
        added_mass_range = [-0.5, 0.5] if RANDOMIZE else [-0.1, 0.1]
        push_robots = RANDOMIZE
        push_interval_s = 5
        max_push_vel_xyz = 0.3
        max_push_avel_xyz = 0.3
        add_action_delay = True
        dynamic_randomization = 0.0
        obs_noise_scale = 0.01



class WalkHWTrainCfg(MimicTrainCfg):
    algorithm_name = 'PPO'

    class runner(MimicTrainCfg.runner):
        run_name = 'walk-hw'
        max_iterations = 4000  # number of policy updates

    class algorithm(MimicTrainCfg.algorithm):

        learning_rate = 3.e-4
        schedule = 'adaptive'

        entropy_coef = 0.01
        value_loss_coef = 0.5
        clip_param = 0.2
        desired_kl = 0.02

        bootstrap = True

    class policy(MimicTrainCfg.policy):

        log_std_init = -1.0
        activation = 'elu'
