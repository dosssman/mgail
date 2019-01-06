import numpy as np
import os
from environment import Environment
from driver import Driver
import logger, time

def dispatcher(env):

    driver = Driver(env)

    while driver.itr < env.n_train_iters:

        # Train
        if env.train_mode:
            driver.train_step()

        # Test
        if driver.itr % env.test_interval == 0:
            print( "### DEBUG: Testing at Iteration %d" % driver.itr)
            # measure performance
            R = []
            for n in range(env.n_episodes_test):
                R.append(driver.collect_experience(record=True, vis=env.vis_flag, noise_flag=False, n_steps=1000))

            # update stats
            driver.reward_mean = sum(R) / len(R)
            driver.reward_std = np.std(R)

            # print info line
            driver.print_info_line('full')

            # TB Logging
            logger.record_tabular("Rew/Mean", driver.reward_mean)
            logger.record_tabular("Rew/Std", driver.reward_std)
            # logger.record_tabular("Rew/Best", driver.best_reward)
            logger.record_tabular("Loss/FwdModel", driver.loss[0])
            logger.record_tabular("Loss/Disc", driver.loss[1])
            logger.record_tabular("Loss/Policy", driver.loss[2])
            logger.record_tabular("Disc/Acc", driver.disc_acc)

            logger.dump_tabular()

            # save snapshot
            if env.train_mode and env.save_models:
                driver.save_model(dir_name=env.config_dir)

        driver.itr += 1


if __name__ == '__main__':
    # load environment
    env = Environment(os.path.curdir, 'Hopper-v1')

    # Tensorboard
    tf_logdir = "tb/"+ time.strftime("%Y-%m-%d-%H-%M-%S")
    logger.configure( tf_logdir, format_strs="stdout,log,csv,tensorboard")
    print( "### DEBUG: Logging to %s" % logger.get_dir())

    # start training
    try:
        dispatcher(env=env)
    except Exception as e:
        raise e
    finally:
        env.gym.close()
