import gym
import gym_qap
import gym_qapConst
import gym_qapImg
import gym_qapImgConst
import tensorflow as tf
import os
import argparse

from stable_baselines import DQN


parser = argparse.ArgumentParser()
parser.add_argument("num_prod", type=float, help="Numero prodotti (deve essere un quadrato)")
parser.add_argument("steps", type=float, help="Numero di steps da effettuare Es.[2e5]")
parser.add_argument("lr", type=float, help="learning_rate Es[1e-4]")
parser.add_argument("model_name", type=str, help="model name to be loaded")
parser.add_argument("env", type=str, help="Nome dell'environment: Const lo stato iniziale e' fisso, \
    Img l'osservazione e' basata sull'immagine ", choices = ['qapConst-v0', 'qapImgConst-v0', 'qap-v0', 'qapImg-v0'])
args = parser.parse_args()

env = gym.make(args.env)

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

# Train the agent
path = os.getenv("HOME")+"/models/"+args.model_name
model = DQN.load(path,tensorboard_log=os.getenv("HOME")+"/tensorboard/")
model.set_env(env)
model.learn(total_timesteps=int(args.steps),callback=callback)
# Save the agent
path = os.getenv("HOME")+"/models/"+args.model_name+"_retrained"
model.save(path)