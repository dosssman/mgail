import tensorflow as tf
import numpy as np
import gym, os
from gym_torcs import TorcsEnv

class Environment(object):
    def __init__(self, run_dir, env_name):
        self.name = env_name
        # self.gym = gym.make(self.name)
        # XXX: Hook up Gym Torcs
        vision = False
        throttle = True
        gear_change = False

        # Agent only
        # race_config_path = os.path.dirname(os.path.abspath(__file__)) + \
        #     "/raceconfig/agent_practice.xml"

        # 2Damned_Agent_2Damned_1Fixed
        # race_config_path = os.path.dirname(os.path.abspath(__file__)) + \
        #     "/raceconfig/2damned_agent_2damned_1fixed.xml"

        # DamDamAgentFix
        # race_config_path = os.path.dirname(os.path.abspath(__file__)) + \
        #     "/raceconfig/2damned_agent_1fixed_record.xml"

        # Agent10Fixed_Track 2 Var 1
        # race_config_path = os.path.dirname(os.path.abspath(__file__)) + \
        #     "/raceconfig/agent_10fixed_sparsed_track_2_var_1.xml"

        # Agent10Fixed_Sparse
        race_config_path = os.path.dirname(os.path.abspath(__file__)) + \
            "/raceconfig/agent_10fixed_sparsed_4.xml"

        rendering = True
        noisy = False

        # TODO: How Restrict to 3 laps when evaling ?
        lap_limiter = 3
        timestep_limit = -1

        # env = gym.make(args.env_id)
        env = TorcsEnv(vision=vision, throttle=True, gear_change=False,
    		race_config_path=race_config_path, rendering=rendering,
    		lap_limiter = lap_limiter, noisy=noisy, timestep_limit=timestep_limit)
        # self.reset()
        env.reset()

        self.gym = env
        self.random_initialization = False
        self._connect()
        self._train_params()
        self.run_dir = run_dir
        self.t = 0

    def _step(self, action):
        action = np.squeeze(action)
        self.t += 1
        result = self.gym.step(action)
        self.state, self.reward, self.done, self.info = result[:4]

        if self.random_initialization:
            self.qpos, self.qvel = self.gym.env.model.data.qpos.flatten(), self.gym.env.model.data.qvel.flatten()
            return np.float32(self.state), np.float32(self.reward), self.done, np.float32(self.qpos), np.float32(self.qvel)
        else:
            return np.float32(self.state), np.float32(self.reward), self.done

    def step(self, action, mode):
        qvel, qpos = [], []
        if mode == 'tensorflow':
            if self.random_initialization:
                state, reward, done, qval, qpos = tf.py_func(self._step, inp=[action], Tout=[tf.float32, tf.float32, tf.bool, tf.float32, tf.float32], name='env_step_func')
            else:
                state, reward, done = tf.py_func(self._step, inp=[action],
                                                 Tout=[tf.float32, tf.float32, tf.bool],
                                                 name='env_step_func')

            state = tf.reshape(state, shape=(self.state_size,))
            done.set_shape(())
        else:
            if self.random_initialization:
                state, reward, done, qvel, qpos = self._step(action)
            else:
                state, reward, done = self._step(action)

        # print("### DEBUG: Done at step() @t=%d" % self.t, done)
        return state, reward, done, 0., qvel, qpos

    def reset(self, relaunch=False, qpos=None, qvel=None):
        self.t = 0
        self.state = self.gym.reset( relaunch=relaunch)
        if self.random_initialization and qpos is not None and qvel is not None:
            self.gym.env.set_state(qpos, qvel)
        return self.state

    def get_status(self):
        return self.done

    def get_state(self):
        return self.state

    def render(self):
        self.gym.render()

    def _connect(self):
        self.state_size = self.gym.observation_space.shape[0]
        self.action_size = self.gym.action_space.shape[0]
        self.action_space = np.asarray([None]*self.action_size)
        # self.qpos_size = self.gym.env.data.qpos.shape[0]
        # self.qvel_size = self.gym.env.data.qvel.shape[0]

    def _train_params(self):
        self.trained_model = None
        self.train_mode = True
        self.expert_data = '/home/z3r0/random/rl/mgail/expert_trajectories/hopper_er.bin'
        self.expert_data_2 = '/home/z3r0/random/rl/openai_logs/openai-gailtorcs/data/Doss10FixedAnal_220eps/expert_data.npz'
        self.n_train_iters = 1000000
        self.n_episodes_test = 1
        self.test_interval = 1000
        self.n_steps_test = 1000
        self.vis_flag = False
        self.save_models = True
        self.config_dir = None
        self.continuous_actions = True

        # Main parameters to play with:
        self.er_agent_size = 1000
        self.prep_time = 1000
        self.collect_experience_interval = 15
        self.n_steps_train = 10
        self.discr_policy_itrvl = 100
        self.gamma = 0.99
        self.batch_size = 70
        self.weight_decay = 1e-7
        self.policy_al_w = 1e-2
        self.policy_tr_w = 1e-4
        self.policy_accum_steps = 7
        self.total_trans_err_allowed = 1000
        self.temp = 1.
        self.cost_sensitive_weight = 0.8
        self.noise_intensity = 1.
        self.do_keep_prob = 0.75

        # Hidden layers size
        self.fm_size = 100
        self.d_size = [200, 100]
        self.p_size = [100, 50]

        # Learning rates
        self.fm_lr = 1e-4
        self.d_lr = 1e-3
        self.p_lr = 1e-4
