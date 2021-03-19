import numpy as np
import os
from environment import Environment
from driver import Driver
import gym_minigrid
import pybulletgym
import matplotlib.pyplot as plt

def plotLoss(losses):
    plt.figure()
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

def plotReward(rewards):
    plt.figure()
    plt.plot(rewards)
    plt.xlabel('Epochs')
    plt.ylabel('Avg Reward')

def dispatcher(env):

    driver = Driver(env)
    avg_rewards = []
    if env.vis_flag:
        env.render()

    while driver.itr < env.n_train_iters:

        # Train
        if env.train_mode:
            driver.train_step()

        # Test
        if driver.itr % env.test_interval == 0:

            # measure performance
            R = []
            for n in range(env.n_episodes_test):
                R.append(driver.collect_experience(record=True, vis=env.vis_flag, noise_flag=True, n_steps=1000))

            # update stats
            driver.reward_mean = sum(R) / len(R)
            driver.reward_std = np.std(R)
            avg_rewards.append(driver.reward_mean)

            # print info line
            driver.print_info_line('full')

            # save snapshot
            if env.train_mode and env.save_models:
                driver.save_model(dir_name=env.config_dir)

        driver.itr += 1

    plotLoss(driver.policy_losses)
    plt.title("Policy Loss")
    plotLoss(driver.disc_losses)
    plt.title("Discriminator Loss")
    plotLoss(driver.forward_losses)
    plt.title("Forward Model Loss")
    plotReward(avg_rewards)
    plt.title("Hopper Average Rewards")
    plt.show()

if __name__ == '__main__':
    # load environment
    env = Environment(os.path.curdir, 'HopperMuJoCoEnv-v0')

    # start training
    dispatcher(env=env)
