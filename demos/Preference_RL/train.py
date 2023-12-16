# RL training based on feature-based trajectory rewards

import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env_name = 'MountainCarContinuous-v0'
gym_env = gym.make(env_name)

np.random.seed(2023)
gym_env.seed(2023)

class TrajectoryRewardEnvironment(gym.Env):
    """A wrapper around an OpenAI Gym environment, which gives a reward based on a given feature function
    of an entire trajectory."""
    def __init__(self, original_env, feature_func, max_episode_length, weights):
        super().__init__()
        self.original_env = original_env
        self.feature_func = feature_func
        self.max_episode_length = max_episode_length
        self.weights = weights

        self.action_space = self.original_env.action_space
        self.observation_space = self.original_env.observation_space

        self.current_trajectory = []
        self.current_length = 0

    def step(self, action):
        # Execute the action in the original environment
        state, _, done, info = self.original_env.step(action)
        self.current_trajectory.append((state, action))

        self.current_length += 1
        reward = 0

        # Check if the episode should end
        if self.current_length >= self.max_episode_length or done:
            feature_vector = self.feature_func(self.current_trajectory)
            reward = np.dot(self.weights, feature_vector)
            self.current_trajectory = []
            self.current_length = 0
            done = True

        return state, reward, done, info

    def reset(self):
        # Reset the original environment and the trajectory
        initial_state = self.original_env.reset()
        self.current_trajectory = []
        self.current_length = 0
        return initial_state

    def render(self, mode='human', close=False):
        return self.original_env.render(mode, close)

    # Implement other necessary methods based on your requirements

def feature_func(traj):
    """Returns the features of the given MountainCar trajectory, i.e. \Phi(traj).
    
    Args:
        traj: List of state-action tuples, e.g. [(state0, action0), (state1, action1), ...]
    
    Returns:
        features: a numpy vector corresponding the features of the trajectory
    """
    states = np.array([pair[0] for pair in traj])
    actions = np.array([pair[1] for pair in traj[:-1]])
    min_pos, max_pos = states[:,0].min(), states[:,0].max()
    mean_speed = np.abs(states[:,1]).mean()
    mean_vec = [-0.703, -0.344, 0.007]
    std_vec = [0.075, 0.074, 0.003]
    return (np.array([min_pos, max_pos, mean_speed]) - mean_vec) / std_vec

trajectory_env = TrajectoryRewardEnvironment(
    gym_env,
    feature_func=feature_func,
    max_episode_length=300,
    weights=np.array([-0.2, 0.7, 0.6]) # insert learnt reward function weights from APREL here
)

# Assuming `env` is your custom environment instance
# For example:
# env = YourCustomMountainCarEnv()
# Replace the above line with the actual instantiation of your environment

# Initialize the agent with your environment instance
model = PPO("MlpPolicy", trajectory_env, verbose=1, learning_rate=1e-3)

# uncommment this line if you want to load a pretrained model
#model = PPO.load("ppo_mountain_car", env=trajectory_env)

try:
    # Train the agent
    model.learn(total_timesteps=int(1e6))  # Adjust the timesteps as needed
except KeyboardInterrupt:
    print("Training interrupted, saving model...")
    model.save("ppo_mountain_car")
    print(f"Model saved to ppo_mountain_car")

# Optionally, you can evaluate the model after the interruption
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
# print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Save the model after training (or interruption)
model.save("ppo_mountain_car")
print(f"Model saved to ppo_mountain_car")


# To load the model, you can use: model = PPO.load("ppo_mountain_car", env=env)
