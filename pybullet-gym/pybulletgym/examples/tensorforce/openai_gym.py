# Copyright 2017 reinforce.io. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
OpenAI gym execution.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import time

import gym.envs
import pybulletgym.envs
from tensorforce import TensorForceError
from tensorforce.agents import Agent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

# Roboschool
# Examples (train)
# python ./openai_gym.py InvertedPendulumPyBulletEnv-v0 -a ./configs/ppo.json -n ./configs/mlp2_network.json -e 1000000 -m 2000 -s ./checkpoints/invpendulumv0-ckpts/invpdv0 -D
# python ./openai_gym.py InvertedDoublePendulumPyBulletEnv-v0 -a ./configs/ppo.json -n ./configs/mlp2_network.json -e 1000000 -m 2000 -s ./checkpoints/invdpendulumv0-ckpts/invdpdv0 -D
# python ./openai_gym.py ReacherPyBulletEnv-v0 -a ./configs/ppo.json -n ./configs/mlp2_network.json -e 1000000 -m 2000 -s ./checkpoints/reacherv0-ckpts/reacherv0 -D
# python ./openai_gym.py AntPyBulletEnv-v0 -a ./configs/ppo.json -n ./configs/mlp2_network.json -e 1000000 -m 2000 -s ./checkpoints/antv0-ckpts/antv0 -D
# python ./openai_gym.py HalfCheetahPyBulletEnv-v0 -a ./configs/ppo.json -n ./configs/mlp2_network.json -e 1000000 -m 2000 -s ./checkpoints/hcheetahv0-ckpts/hcheetahv0 -D
# python ./openai_gym.py HumanoidPyBulletEnv-v0 -a ./configs/ppo.json -n ./configs/mlp2_network.json -e 1000000 -m 2000 -s ./checkpoints/humanoidv0-ckpts/humanoidv0 -D
# python ./openai_gym.py PusherPyBulletEnv-v0 -a ./configs/ppo.json -n ./configs/mlp2_network.json -e 1000000 -m 2000 -s ./checkpoints/pusherv0-ckpts/pusherv0 -D

# Examples (test)
# python ./openai_gym.py InvertedPendulumPyBulletEnv-v0 -a ./configs/ppo.json -n ./configs/mlp2_network.json -e 1000000 -m 2000 -l ./checkpoints/invpendulumv0-ckpts/ -D --test --visualize
# python ./openai_gym.py InvertedDoublePendulumPyBulletEnv-v0 -a ./configs/ppo.json -n ./configs/mlp2_network.json -e 1000000 -m 2000 -l ./checkpoints/invdpendulumv0-ckpts/ -D --test --visualize
# python ./openai_gym.py ReacherPyBulletEnv-v0 -a ./configs/ppo.json -n ./configs/mlp2_network.json -e 1000000 -m 2000 -l ./checkpoints/reacherv0-ckpts/ -D --test --visualize
# python ./openai_gym.py AntPyBulletEnv-v0 -a ./configs/ppo.json -n ./configs/mlp2_network.json -e 1000000 -m 2000 -l ./checkpoints/antv0-ckpts/ -D --test --visualize
# python ./openai_gym.py HalfCheetahPyBulletEnv-v0 -a ./configs/ppo.json -n ./configs/mlp2_network.json -e 1000000 -m 2000 -l ./checkpoints/hcheetahv0-ckpts/ -D --test --visualize
# python ./openai_gym.py HumanoidPyBulletEnv-v0 -a ./configs/ppo.json -n ./configs/mlp2_network.json -e 1000000 -m 2000 -l ./checkpoints/humanoidv0-ckpts/ -D --test --visualize
# python ./openai_gym.py PusherPyBulletEnv-v0 -a ./configs/ppo.json -n ./configs/mlp2_network.json -e 1000000 -m 2000 -l ./checkpoints/pusherv0-ckpts/ -D --test --visualize

# MuJoCo
# Examples (train)
# python ./openai_gym.py InvertedPendulumPyBulletEnv-v0 -a ./configs/ppo.json -n ./configs/mlp2_network.json -e 1000000 -m 2000 -s ./checkpoints/mujoco-invpendulumv0/invpdv0 -D
# python ./openai_gym.py InvertedDoublePendulumPyBulletEnv-v0 -a ./configs/ppo.json -n ./configs/mlp2_network.json -e 1000000 -m 2000 -s ./checkpoints/mujoco-invdpendulumv0/invdpdv0 -D
# python ./openai_gym.py ReacherPyBulletEnv-v0 -a ./configs/ppo.json -n ./configs/mlp2_network.json -e 1000000 -m 2000 -s ./checkpoints/mujoco-reacherv0/reacherv0 -D
# python ./openai_gym.py HopperMuJoCoEnv-v0 -a ./configs/ppo.json -n ./configs/mlp2_network.json -e 1000000 -m 2000 -s ./checkpoints/mujoco-hopperv0/hopperv0 -D
# python ./openai_gym.py AntPyBulletEnv-v0 -a ./configs/ppo.json -n ./configs/mlp2_network.json -e 1000000 -m 2000 -s ./checkpoints/mujoco-antv0/antv0 -D
# python ./openai_gym.py HalfCheetahPyBulletEnv-v0 -a ./configs/ppo.json -n ./configs/mlp2_network.json -e 1000000 -m 2000 -s ./checkpoints/mujoco-halfcheetahv0/halfcheetahv0 -D
# python ./openai_gym.py HumanoidPyBulletEnv-v0 -a ./configs/ppo.json -n ./configs/mlp2_network.json -e 1000000 -m 2000 -s ./checkpoints/mujoco-humanoidv0/humanoidv0 -D

# Examples (test)
# python ./openai_gym.py InvertedPendulumMuJoCoEnv-v0 -a ./configs/ppo.json -n ./configs/mlp2_network.json -e 1000000 -m 2000 -l ./checkpoints/mujoco-invpendulumv2/ -D --test --visualize
# python ./openai_gym.py InvertedDoublePendulumMuJoCoEnv-v0 -a ./configs/ppo.json -n ./configs/mlp2_network.json -e 1000000 -m 2000 -l ./checkpoints/mujoco-invdpendulumv2/ -D --test --visualize
# python ./openai_gym.py ReacherMuJoCoEnv-v0 -a ./configs/ppo.json -n ./configs/mlp2_network.json -e 1000000 -m 2000 -l ./checkpoints/mujoco-reacherv2/ -D --test --visualize
# python ./openai_gym.py AntMuJoCoEnv-v0 -a ./configs/ppo.json -n ./configs/mlp2_network.json -e 1000000 -m 2000 -l ./checkpoints/mujoco-antv2/ -D --test --visualize
# python ./openai_gym.py HalfCheetahMuJoCoEnv-v0 -a ./configs/ppo.json -n ./configs/mlp2_network.json -e 1000000 -m 2000 -l ./checkpoints/mujoco-halfcheetahv2/ -D --test --visualize
# python ./openai_gym.py HumanoidMuJoCoEnv-v0 -a ./configs/ppo.json -n ./configs/mlp2_network.json -e 1000000 -m 2000 -l ./checkpoints/mujoco-humanoidv2/ -D --test --visualize
# python ./openai_gym.py PusherMuJoCoEnv-v0 -a ./configs/ppo.json -n ./configs/mlp2_network.json -e 1000000 -m 2000 -l ./checkpoints/mujoco-pusherv2/ -D --test --visualize

# For detailed explanations about the options, check the arguments implementation below or on the tensorforce main repository: https://github.com/reinforceio/tensorforce


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('gym_id', help="Id of the Gym environment")
    parser.add_argument('-a', '--agent', help="Agent configuration file")
    parser.add_argument('-n', '--network', default=None, help="Network specification file")
    parser.add_argument('-e', '--episodes', type=int, default=None, help="Number of episodes")
    parser.add_argument('-t', '--timesteps', type=int, default=None, help="Number of timesteps")
    parser.add_argument('-m', '--max-episode-timesteps', type=int, default=None, help="Maximum number of timesteps per episode")
    parser.add_argument('-d', '--deterministic', action='store_true', default=False, help="Choose actions deterministically")
    parser.add_argument('-s', '--save', help="Save agent to this dir")
    parser.add_argument('-se', '--save-episodes', type=int, default=100, help="Save agent every x episodes")
    parser.add_argument('-l', '--load', help="Load agent from this dir")
    parser.add_argument('--monitor', help="Save results to this directory")
    parser.add_argument('--monitor-safe', action='store_true', default=False, help="Do not overwrite previous results")
    parser.add_argument('--monitor-video', type=int, default=0, help="Save video every x steps (0 = disabled)")
    parser.add_argument('--visualize', action='store_true', default=False, help="Enable OpenAI Gym's visualization")
    parser.add_argument('-D', '--debug', action='store_true', default=False, help="Show debug outputs")
    parser.add_argument('-te', '--test', action='store_true', default=False, help="Test agent without learning.")
    parser.add_argument('--job', type=str, default=None, help="For distributed mode: The job type of this agent.")
    parser.add_argument('--task', type=int, default=0, help="For distributed mode: The task index of this agent.")
    parser.add_argument('--sleep', type=float, default=None, help='To make the simulation slower for analysis.')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)

    environment = OpenAIGym(
        gym_id=args.gym_id,
        monitor=args.monitor,
        monitor_safe=args.monitor_safe,
        monitor_video=args.monitor_video,
        visualize=args.visualize
    )

    # initialize visualization
    if args.visualize:
        environment.gym.render(mode="human") # HACK to get the visualizer started

    if args.agent is not None:
        with open(args.agent, 'r') as fp:
            agent = json.load(fp=fp)
    else:
        raise TensorForceError("No agent configuration provided.")

    if args.network is not None:
        with open(args.network, 'r') as fp:
            network = json.load(fp=fp)
    else:
        network = None
        logger.info("No network configuration provided.")

    # TEST
    agent["execution"] = dict(
        type="distributed",
        distributed_spec=dict(
            job=args.job,
            task_index=args.task,
            # parameter_server=(args.job == "ps"),
            cluster_spec=dict(
                ps=["192.168.2.107:22222"],
                worker=["192.168.2.107:22223"]
            ))
    ) if args.job else None
    # END: TEST

    agent = Agent.from_spec(
        spec=agent,
        kwargs=dict(
            states=environment.states,
            actions=environment.actions,
            network=network,
        )
    )
    if args.load:
        load_dir = os.path.dirname(args.load)
        if not os.path.isdir(load_dir):
            raise OSError("Could not load agent from {}: No such directory.".format(load_dir))
        agent.restore_model(args.load)

    if args.save:
        save_dir = os.path.dirname(args.save)
        if not os.path.isdir(save_dir):
            try:
                os.mkdir(save_dir, 0o755)
            except OSError:
                raise OSError("Cannot save agent to dir {} ()".format(save_dir))

    if args.debug:
        logger.info("-" * 16)
        logger.info("Configuration:")
        logger.info(agent)

    runner = Runner(
        agent=agent,
        environment=environment,
        repeat_actions=1
    )

    if args.debug:  # TODO: Timestep-based reporting
        report_episodes = 1
    else:
        report_episodes = 100

    logger.info("Starting {agent} for Environment '{env}'".format(agent=agent, env=environment))

    def episode_finished(r, id_):
        if r.episode % report_episodes == 0:
            steps_per_second = r.timestep / (time.time() - r.start_time)
            logger.info("Finished episode {:d} after {:d} timesteps. Steps Per Second {:0.2f}".format(
                r.agent.episode, r.episode_timestep, steps_per_second
            ))
            logger.info("Episode reward: {}".format(r.episode_rewards[-1]))
            logger.info("Average of last 500 rewards: {:0.2f}".
                        format(sum(r.episode_rewards[-500:]) / min(500, len(r.episode_rewards))))
            logger.info("Average of last 100 rewards: {:0.2f}".
                        format(sum(r.episode_rewards[-100:]) / min(100, len(r.episode_rewards))))
        if args.save and args.save_episodes is not None and not r.episode % args.save_episodes:
            logger.info("Saving agent to {}".format(args.save))
            r.agent.save_model(args.save)

        return True

    runner.run(
        num_timesteps=args.timesteps,
        num_episodes=args.episodes,
        max_episode_timesteps=args.max_episode_timesteps,
        deterministic=args.deterministic,
        episode_finished=episode_finished,
        testing=args.test,
        sleep=args.sleep
    )
    runner.close()

    logger.info("Learning finished. Total episodes: {ep}".format(ep=runner.agent.episode))


if __name__ == '__main__':
    main()
