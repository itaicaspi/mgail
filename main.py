import os
from environment import Environment
from dispatcher import dispatcher


class EnvironmentName:
    HOPPER = 'Hopper-v1'
    WALKER = 'Walker2d-v1'
    HALF_CHEETAH = 'HalfCheetah-v1'
    ANT = 'Ant-v1'
    MOUNTAIN_CAR = 'MountainCar-v0'
    HUMANOID = 'Humanoid-v1'

if __name__ == '__main__':
    # load environment
    env = Environment(os.path.curdir, EnvironmentName.HOPPER)

    # start training
    dispatcher(env=env)
