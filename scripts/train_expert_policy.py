#!/usr/bin/env python3
import gym
from gym_minigrid.wrappers import *
from garage import wrap_experiment
from garage.envs import GarageEnv
from garage.experiment import LocalTFRunner
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import TRPO
from garage.tf.policies import CategoricalMLPPolicy

from akro.discrete import Discrete

# These wrappers are inspired by https://github.com/maximecb/gym-minigrid/blob/master/gym_minigrid/wrappers.py
class OnlyPartialObjAndColor(gym.core.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(7 * 7 * 2,),
            dtype='uint8'
        )

    def observation(self, obs):
        return np.delete(obs["image"], 2, axis=2).flatten()

MAX_STEPS = 19 ** 2 * 2

@wrap_experiment
def trpo_minigrid(ctxt=None, seed=1):
    """Train TRPO with MiniGrid-FourRooms-v0 environment.
    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
    """
    set_seed(seed)
    with LocalTFRunner(ctxt) as runner:
        env = gym.make("MiniGrid-FourRooms-v0")
        # The modification of the env has to be made before using wrapper
        env.action_space = Discrete(3)
        env.max_steps = MAX_STEPS
        env = OnlyPartialObjAndColor(env)
        env = GarageEnv(env)
        
        policy = CategoricalMLPPolicy(name='policy',
                                      env_spec=env.spec,
                                      hidden_sizes=(128, 64, 32))

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    discount=0.99,
                    max_kl_step=0.001, # 0.001 is better than 0.01
                    max_path_length=MAX_STEPS,)

        runner.setup(algo, env)
        runner.train(n_epochs=2000, batch_size=4000)


trpo_minigrid()
