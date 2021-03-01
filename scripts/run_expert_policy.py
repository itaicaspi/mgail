from garage.experiment import Snapshotter
import tensorflow as tf

MODEL_PATH = "data/local/experiment/trpo_minigrid/"

def main():
    snapshotter = Snapshotter()
    with tf.compat.v1.Session():
        print("loading model...")
        data = snapshotter.load(MODEL_PATH)
        print("model", data)
        policy = data['algo'].policy
        env = data['env']

        steps, max_steps = 0, 1500
        done = False
        obs = env.reset()  # The initial observation
        policy.reset()


        while steps < max_steps and not done:
            action = policy.get_action(obs)[0]
            obs, rew, done, _ = env.step(action)
            env.render()  # Render the environment to see what's going on (optional)
            steps += 1

        env.close()

if __name__ == "__main__":
    main()
