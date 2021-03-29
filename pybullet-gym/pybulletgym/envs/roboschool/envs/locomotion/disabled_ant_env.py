from pybulletgym.envs.roboschool.envs.locomotion.walker_base_env import WalkerBaseBulletEnv
from pybulletgym.envs.roboschool.robots.locomotors import DisabledAnt


class DisabledAntBulletEnv(WalkerBaseBulletEnv):
    def __init__(self):
        self.robot = DisabledAnt()
        WalkerBaseBulletEnv.__init__(self, self.robot)
