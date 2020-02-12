import gym
import gym_qap
import gym_qapImg
import tensorflow as tf



from stable_baselines import DQN
from stable_baselines import A2C
from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy

env = gym.make('qap-v0')

num_steps = 0

def callback(locals_, globals_):
    self_ = locals_['self']
    global num_steps
    if (num_steps+1) % 1000 == 0:
    	current_impr = (env.initial_sum-env.current_sum)/env.initial_sum*100
    #mff_impr =(env.initial_sum-env.mff_sum)/env.initial_sum*100
    #value = current_impr - mff_impr
    	summary = tf.Summary(value=[tf.Summary.Value(tag='improvement', simple_value=current_impr)])
    	locals_['writer'].add_summary(summary, self_.num_timesteps)
    num_steps+=1
    return True

# Instantiate the agent
model = DQN('MlpPolicy', env, verbose=0,tensorboard_log="./tensorboard/")
#model = DQN('MlpPolicy', env, learning_rate=1e-3, prioritized_replay=True, verbose=1,tensorboard_log="./tensorboard/")
# Train the agent
model.learn(total_timesteps=int(4e5),callback=callback)
# Save the agent
model.save("./models/dqn10Test")

