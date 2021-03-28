from garage.experiment import Snapshotter
import tensorflow as tf # optional, only for TensorFlow as we need a tf.Session
from garage.sampler.utils import rollout

import numpy as np
import gzip
import h5py
import argparse



MODEL_PATH = "data/local/experiment/antbullet/"

# MAIN IDEAS: 
# 1. https://github.com/rlworkgroup/garage/blob/c43eaf7647f7feb467847cb8bc107301a7c31938/docs/user/reuse_garage_policy.md
# 2. https://github.com/rail-berkeley/d4rl/blob/master/scripts/generation/generate_minigrid_fourroom_data.py

def reset_data():
    return {'observations': [],
            'actions': [],
            'terminals': [],
            'rewards': [],
            'a_info/mean': [],
            'a_info/log_std': [],
            }

def append_data(data, s, a, a_info, done, rew):
    data['observations'].append(s)
    data['actions'].append(a)
    data['rewards'].append(rew)
    data['terminals'].append(done)
    data['a_info/mean'].append(a_info['mean'])
    data['a_info/log_std'].append(a_info['log_std'])

def npify(data):
    for k in data:
        if k == 'terminals':
            dtype = np.bool_
        else:
            dtype = np.float32
        data[k] = np.array(data[k], dtype=dtype)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='Render trajectories')
    parser.add_argument('--random', action='store_true', help='Noisy actions')
    parser.add_argument('--num_samples', type=int, default=int(1e5), help='Num samples to collect')
    args = parser.parse_args()


    buffer_data = reset_data()
    snapshotter = Snapshotter()

    with tf.compat.v1.Session(): # optional, only for TensorFlow
        data = snapshotter.load(MODEL_PATH)
        policy = data['algo'].policy
        env = data['env']

        steps, max_steps = 0, 1500
        if args.render:
            env.render()
        obs = env.reset()  # The initial observation
        policy.reset()
        done = False
        ts = 0

        for _ in range(args.num_samples):
            obs = env.reset()  # The initial observation
            policy.reset()
            done = False
            rew = 0.0
            ts = 0
            # if _ % 1000 == 0:
            print('episode: ', _)
            
            for _ in range(max_steps):
                if args.render:
                    env.render()

                act, prob = policy.get_action(obs)
                
                # act[0] is the actual action, while the second tuple is the done variable. Inspiration: 
                # https://github.com/lcswillems/rl-starter-files/blob/3c7289765883ca681e586b51acf99df1351f8ead/utils/agent.py#L47

                append_data(buffer_data, obs, act, prob, done, rew)
                new_obs, rew, done, _ = env.step(act) # why [0] ?
                ts += 1

                if done: 
                    # reset target here!
                    random_act = env.action_space.sample()
                    infos = {'mean': np.random.rand(env.action_space.shape[0]), 'log_std': np.random.rand(env.action_space.shape[0])} # random action info
                    append_data(buffer_data, new_obs, random_act, infos , done, rew)
                    break

                else:
                    # continue by setting current obs
                    obs = new_obs

        fname = 'generated_antbullet_probs.hdf5'
        dataset = h5py.File(fname, 'w')
        npify(buffer_data)
        for key in buffer_data:
            dataset.create_dataset(key, data=buffer_data[key], compression='gzip')

        env.close()

if __name__ == "__main__":
    main()




