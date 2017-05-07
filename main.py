import sys
import os
from environment import Environment

git_path = '/home/llt_lab/Documents/'

# sys.path.append(git_path + '/Buffe/utils')
sys.path.append(git_path + '/Buffe/Applications/mgail/')

from dispatcher import dispatcher

class EnvironmentName:
    LINEMOVE_2D = 'linemove_2D'
    HOPPER = 'Hopper-v1'
    WALKER = 'Walker2d-v1'
    HALFCHEETAH = 'HalfCheetah-v1'
    ANT = 'Ant-v1'
    MOUNTAINCAR = 'MountainCar-v0'
    HUMANOID = 'Humanoid-v1'

if __name__ == '__main__':

    env_name = EnvironmentName.HOPPER
    run_dir = git_path + '/Buffe/Applications/mgail/'
    env = Environment(run_dir, env_name)

    dispatcher(env=env)
