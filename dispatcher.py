import time
import numpy as np
import common
from driver import Driver


def dispatcher(env):

    print '... Building controller'
    t0 = time.clock()

    if env.expert_data is None:
        env.record_expert()
        return

    driver = Driver(env)

    if env.train_mode:
        print 'Built controller in %0.2f [min]\n ... Training controller' % ((time.clock()-t0)/60)
    else:
        print 'Built controller in %0.2f [min]\n ... Playing saved model %s' % ((time.clock()-t0)/60, env.trained_model)

    while driver.itr < env.n_train_iters:

        # Train
        if env.train_mode:
            driver.train_step()

        # Test
        if driver.itr % env.test_interval == 0:

            # measure performance
            R = []
            for n in range(env.n_episodes_test):
                R.append(driver.collect_experience(record=True, vis=env.vis_flag, noise_flag=False, n_steps=1000))

            # update stats
            driver.reward_mean = sum(R) / len(R)
            driver.reward_std = np.std(R)

            # print info line
            driver.print_info_line('full')

            # save experience buffer
            if driver.env.save_agent_er and driver.itr > driver.env.save_agent_at_itr:
                common.save_er(directory=driver.env.run_dir, module=driver.algorithm.er_agent, exit_=True)

            # save snapshot
            if env.train_mode and (env.save_models or driver.reward_mean > env.good_reward):
                driver.save_model(dir_name=env.config_dir)

        driver.itr += 1
