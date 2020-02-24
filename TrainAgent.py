import gym
import gym_qap
import gym_qapConst
import gym_qapImg
import tensorflow as tf
import os


from stable_baselines import DQN
from stable_baselines import A2C
from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy

env = gym.make('qapConst-v0')

num_step = 0
def callback(locals_, globals_):
    self_ = locals_['self']
    global num_step
    if env.done:
        final_impr = (env.initial_sum-env.final_sum)/env.initial_sum*100
        mff_impr =(env.initial_sum-env.mff_sum)/env.initial_sum*100
        over_mff = final_impr - mff_impr
        summary = tf.Summary(value=[tf.Summary.Value(tag='improvement', simple_value=final_impr)])
        summary2 = tf.Summary(value=[tf.Summary.Value(tag='improvement over mff', simple_value=over_mff)])
        locals_['writer'].add_summary(summary, self_.num_timesteps)
        locals_['writer'].add_summary(summary2, self_.num_timesteps)
    num_step+=1
    return True

# Instantiate the agent
#model = A2C('MlpPolicy', env, verbose=0,tensorboard_log="./tensorboard/")
model = DQN('MlpPolicy', env, learning_rate=1e-2, prioritized_replay=True, verbose=0,tensorboard_log=os.getenv("HOME")+"/tensorboard/")
# Train the agent
model.learn(total_timesteps=int(7e5),callback=callback)
# Save the agent
path = os.getenv("HOME")+"/models/model_dqn_const_10_700k"
model.save(path)