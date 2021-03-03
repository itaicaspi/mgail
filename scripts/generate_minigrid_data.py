from garage.experiment import Snapshotter
import tensorflow as tf # optional, only for TensorFlow as we need a tf.Session
from garage.sampler.utils import rollout

import numpy as np
import gzip
import h5py
import argparse



MODEL_PATH = "data/local/experiment/trpo_minigrid/"

# MAIN IDEA: https://github.com/rlworkgroup/garage/blob/c43eaf7647f7feb467847cb8bc107301a7c31938/docs/user/reuse_garage_policy.md

def reset_data():
    return {'observations': [],
            'actions': [],
            'terminals': [],
            'rewards': [],
            'infos/goal': [],
            'infos/pos': [],
            'infos/orientation': [],
            }

def append_data(data, s, a, tgt, done, pos, ori):
    data['observations'].append(s)
    data['actions'].append(a)
    data['rewards'].append(0.0)
    data['terminals'].append(done)
    data['infos/goal'].append(tgt)
    data['infos/pos'].append(pos)
    data['infos/orientation'].append(ori)

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
        obs = env.reset()  # The initial observation
        policy.reset()
        done = False
        ts = 0

        for _ in range(args.num_samples):
            obs = env.reset()  # The initial observation
            policy.reset()
            done = False
            ts = 0
            # print('episode: ', _)
            
            for _ in range(max_steps):
                if args.render:
                    env.render()  # Render the environment to see what's going on (optional)

                act = policy.get_action(obs)

                # if ts >= :
                #     done = True # this forces us to have a terminal episode so it doesn't go on forever
                
                # act[0] is the actual action, while the second tuple is the done variable. Inspiration: 
                # https://github.com/lcswillems/rl-starter-files/blob/3c7289765883ca681e586b51acf99df1351f8ead/utils/agent.py#L47
                # print('shape of action:', act[0], 'dim 1', act[1])
                append_data(buffer_data, obs['image'], act[0], _, done, _, env.agent_dir)
                new_obs, rew, done, _ = env.step(act[0]) # why [0] ?
                ts += 1

                if done: 
                    # reset target here!
                    # obs = env.reset()
                    # done = False
                    # ts = 0
                    break

                else:
                    # continue by setting current obs
                    obs = new_obs

        fname = 'minigrid4rooms_generated.hdf5' 
        dataset = h5py.File(fname, 'w')
        npify(buffer_data)
        for key in buffer_data:
            dataset.create_dataset(key, data=buffer_data[key], compression='gzip')

        env.close()

if __name__ == "__main__":
    main()




