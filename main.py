import os
from environment import Environment
from dispatcher import dispatcher

if __name__ == '__main__':
    # load environment
    env = Environment(os.path.curdir, 'Hopper-v1')

    # start training
    dispatcher(env=env)
