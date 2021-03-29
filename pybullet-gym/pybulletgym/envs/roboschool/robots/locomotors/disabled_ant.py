# from pybulletgym.envs.roboschool.robots.locomotors.walker_base import WalkerBase
# from pybulletgym.envs.roboschool.robots.robot_bases import MJCFBasedRobot

# from pybulletgym.envs.assets.model_builder import MJCModel

# def angry_ant_crippled(gear=150):
#     mjcmodel = MJCModel('ant_maze')
#     mjcmodel.root.compiler(inertiafromgeom="true", angle="degree", coordinate="local")
#     mjcmodel.root.option(timestep="0.01", integrator="RK4")
#     mjcmodel.root.custom().numeric(data="0.0 0.0 0.55 1.0 0.0 0.0 0.0 0.0 1.0 0.0 -1.0 0.0 -1.0 0.0 1.0",name="init_qpos")
#     asset = mjcmodel.root.asset()
#     asset.texture(builtin="gradient",height="100",rgb1="1 1 1",rgb2="0 0 0",type="skybox",width="100")
#     asset.texture(builtin="flat",height="1278",mark="cross",markrgb="1 1 1",name="texgeom",random="0.01",rgb1="0.8 0.6 0.4",rgb2="0.8 0.6 0.4",type="cube",width="127")
#     asset.texture(builtin="checker",height="100",name="texplane",rgb1="0 0 0",rgb2="0.8 0.8 0.8",type="2d",width="100")
#     asset.material(name="MatPlane",reflectance="0.5",shininess="1",specular="1",texrepeat="60 60",texture="texplane")
#     asset.material(name="geom",texture="texgeom",texuniform="true")



#     default = mjcmodel.root.default()
#     default.joint(armature=1, damping=1, limited='true')
#     default.geom(friction=[1.5,0.5,0.5], density=5.0, margin=0.01, condim=3, conaffinity=0, rgba="0.8 0.6 0.4 1")

#     worldbody = mjcmodel.root.worldbody()

#     worldbody.geom(conaffinity=1, condim=3, material="MatPlane",name="floor",pos="0 0 0",rgba="0.8 0.9 0.8 1",size="40 40 40",type="plane")
#     worldbody.light(cutoff="100",diffuse=[.8,.8,.8],dir="-0 0 -1.3",directional="true",exponent="1",pos="0 0 1.3",specular=".1 .1 .1")


#     ant = worldbody.body(name='torso', pos=[0, 0, 0.75])
#     ant.geom(name='torso_geom', pos=[0, 0, 0], size="0.25", type="sphere")
#     ant.joint(armature="0", damping="0", limited="false", margin="0.01", name="root", pos=[0, 0, 0], type="free")

#     eye_z = 0.1
#     eye_y = -.21
#     eye_x_offset = 0.07
#     # eyes
#     ant.geom(fromto=[eye_x_offset,0,eye_z,eye_x_offset,eye_y,eye_z], name='eye1', size='0.03', type='capsule', rgba=[1,1,1,1])
#     ant.geom(fromto=[eye_x_offset,0,eye_z,eye_x_offset,eye_y-0.02,eye_z], name='eye1_', size='0.02', type='capsule', rgba=[0,0,0,1])
#     ant.geom(fromto=[-eye_x_offset,0,eye_z,-eye_x_offset,eye_y,eye_z], name='eye2', size='0.03', type='capsule', rgba=[1,1,1,1])
#     ant.geom(fromto=[-eye_x_offset,0,eye_z,-eye_x_offset,eye_y-0.02,eye_z], name='eye2_', size='0.02', type='capsule', rgba=[0,0,0,1])
#     # eyebrows
#     ant.geom(fromto=[eye_x_offset-0.03,eye_y, eye_z+0.07, eye_x_offset+0.03, eye_y, eye_z+0.1], name='brow1', size='0.02', type='capsule', rgba=[0,0,0,1])
#     ant.geom(fromto=[-eye_x_offset+0.03,eye_y, eye_z+0.07, -eye_x_offset-0.03, eye_y, eye_z+0.1], name='brow2', size='0.02', type='capsule', rgba=[0,0,0,1])




#     front_left_leg = ant.body(name="front_left_leg", pos=[0, 0, 0])
#     front_left_leg.geom(fromto=[0.0, 0.0, 0.0, 0.2, 0.2, 0.0], name="aux_1_geom", size="0.08", type="capsule")
#     aux_1 = front_left_leg.body(name="aux_1", pos=[0.2, 0.2, 0])
#     aux_1.joint(axis=[0, 0, 1], name="hip_1", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge")
#     aux_1.geom(fromto=[0.0, 0.0, 0.0, 0.2, 0.2, 0.0], name="left_leg_geom", size="0.08", type="capsule")
#     ankle_1 = aux_1.body(pos=[0.2, 0.2, 0])
#     ankle_1.joint(axis=[-1, 1, 0], name="ankle_1", pos=[0.0, 0.0, 0.0], range=[30, 70], type="hinge")
#     ankle_1.geom(fromto=[0.0, 0.0, 0.0, 0.4, 0.4, 0.0], name="left_ankle_geom", size="0.08", type="capsule")

#     front_right_leg = ant.body(name="front_right_leg", pos=[0, 0, 0])
#     front_right_leg.geom(fromto=[0.0, 0.0, 0.0, -0.2, 0.2, 0.0], name="aux_2_geom", size="0.08", type="capsule")
#     aux_2 = front_right_leg.body(name="aux_2", pos=[-0.2, 0.2, 0])
#     aux_2.joint(axis=[0, 0, 1], name="hip_2", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge")
#     aux_2.geom(fromto=[0.0, 0.0, 0.0, -0.2, 0.2, 0.0], name="right_leg_geom", size="0.08", type="capsule")
#     ankle_2 = aux_2.body(pos=[-0.2, 0.2, 0])
#     ankle_2.joint(axis=[1, 1, 0], name="ankle_2", pos=[0.0, 0.0, 0.0], range=[-70, -30], type="hinge")
#     ankle_2.geom(fromto=[0.0, 0.0, 0.0, -0.4, 0.4, 0.0], name="right_ankle_geom", size="0.08", type="capsule")

#     # Back left leg is crippled
#     thigh_length = 0.1 #0.2
#     ankle_length = 0.2 #0.4
#     dark_red = [0.8,0.3,0.3,1.0]

#     back_left_leg = ant.body(name="back_left_leg", pos=[0, 0, 0])
#     back_left_leg.geom(fromto=[0.0, 0.0, 0.0, -0.2, -0.2, 0.0], name="aux_3_geom", size="0.08", type="capsule",
#                        rgba=dark_red)
#     aux_3 = back_left_leg.body(name="aux_3", pos=[-0.2, -0.2, 0])
#     aux_3.joint(axis=[0, 0, 1], name="hip_3", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge")
#     aux_3.geom(fromto=[0.0, 0.0, 0.0, -thigh_length, -thigh_length, 0.0], name="backleft_leg_geom", size="0.08", type="capsule",
#                rgba=dark_red)
#     ankle_3 = aux_3.body(pos=[-thigh_length, -thigh_length, 0])
#     ankle_3.joint(axis=[-1, 1, 0], name="ankle_3", pos=[0.0, 0.0, 0.0], range=[-70, -30], type="hinge")
#     ankle_3.geom(fromto=[0.0, 0.0, 0.0, -ankle_length, -ankle_length, 0.0], name="backleft_ankle_geom", size="0.08", type="capsule",
#                  rgba=dark_red)

#     back_right_leg = ant.body(name="back_right_leg", pos=[0, 0, 0])
#     back_right_leg.geom(fromto=[0.0, 0.0, 0.0, 0.2, -0.2, 0.0], name="aux_4_geom", size="0.08", type="capsule",
#                         rgba=dark_red)
#     aux_4 = back_right_leg.body(name="aux_4", pos=[0.2, -0.2, 0])
#     aux_4.joint(axis=[0, 0, 1], name="hip_4", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge")
#     aux_4.geom(fromto=[0.0, 0.0, 0.0, thigh_length, -thigh_length, 0.0], name="backright_leg_geom", size="0.08", type="capsule",
#                rgba=dark_red)
#     ankle_4 = aux_4.body(pos=[thigh_length, -thigh_length, 0])
#     ankle_4.joint(axis=[1, 1, 0], name="ankle_4", pos=[0.0, 0.0, 0.0], range=[30, 70], type="hinge")
#     ankle_4.geom(fromto=[0.0, 0.0, 0.0, ankle_length, -ankle_length, 0.0], name="backright_ankle_geom", size="0.08", type="capsule",
#                  rgba=dark_red)

#     actuator = mjcmodel.root.actuator()
#     actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_1", gear=gear)
#     actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_1", gear=gear)
#     actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_2", gear=gear)
#     actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_2", gear=gear)
#     actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_3", gear=1) # cripple the joints
#     actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_3", gear=1) # cripple the joints
#     actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_4", gear=1)
#     actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_4", gear=1)
#     return mjcmodel

# class DisabledAnt(WalkerBase, MJCFBasedRobot):
#     foot_list = ['front_left_foot', 'front_right_foot', 'left_back_foot', 'right_back_foot']

#     def __init__(self):
#         # model = angry_ant_crippled() # THIS BUILDS THE XML

#         # IF YOU NEED TO SAVE IT TO A FILE FIRST:
#         # with model.asfile() as f:
#         #     print(f.read())  # prints a dump of the model to STDOUT
#         #####
        
#         WalkerBase.__init__(self, power=2.5)
#         MJCFBasedRobot.__init__(self, 'disabled_ant.xml', "torso", action_dim=8, obs_dim=28)


#     def alive_bonus(self, z, pitch):
#         return +1 if z > 0.26 else -1  # 0.25 is central sphere rad, die if it scrapes the ground

from pybulletgym.envs.roboschool.robots.locomotors.walker_base import WalkerBase
from pybulletgym.envs.roboschool.robots.robot_bases import MJCFBasedRobot


class DisabledAnt(WalkerBase, MJCFBasedRobot):
    foot_list = ['front_left_foot', 'front_right_foot', 'left_back_foot', 'right_back_foot']

    def __init__(self):
        WalkerBase.__init__(self, power=2.5)
        MJCFBasedRobot.__init__(self, "disabled_ant.xml", "torso", action_dim=8, obs_dim=28)

    def alive_bonus(self, z, pitch):
        return +1 if z > 0.26 else -1  # 0.25 is central sphere rad, die if it scrapes the ground
