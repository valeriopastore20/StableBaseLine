from stable_baselines.common.env_checker import check_env
import gym
import gym_qap
import gym_qapImg


from stable_baselines import DQN
from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy

env = gym.make('qapImg-v0')
# It will check your custom environment and output additional warnings if needed
check_env(env)


#If you want to quickly try a random agent on your environment, you can also do:

obs = env.reset()
n_steps = 10
for _ in range(n_steps):
    # Random action
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

print("DONE")