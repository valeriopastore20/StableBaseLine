import gym
import gym_qap
import gym_qapConst
import gym_qapImg
import gym_qapImgConst
import tensorflow as tf
import os
import argparse
from stable_baselines.deepq.policies import FeedForwardPolicy
from stable_baselines import DQN

parser = argparse.ArgumentParser()
parser.add_argument("num_prod", type=int, help = "Numero prodotti (deve essere un quadrato)")
parser.add_argument("steps", type=float, help = "Numero di steps da effettuare Es.[2e5]")
parser.add_argument("lr", type=float, help = "learning_rate Es[1e-4]")
parser.add_argument("model_name", type=str, help = "nome del modello da salvare o caricare")
parser.add_argument("retrain", type=str, help = "Specifica se il modello viene allenato da zero o riallenato da uno gia' esistente",\
	choices = ["retrain","no_retrain"])
parser.add_argument("env", type=str, help = "Nome dell'environment: Const lo stato iniziale e' fisso, \
    Img l'osservazione e' basata sull'immagine ", choices = ['qapConst-v0', 'qapImgConst-v0', 'qap-v0', 'qapImg-v0'])
parser.add_argument("policy", type=str, help = "Specifica se deve essere usata un'architettur di default oppure \
	una custom (Ovviamente ha effetto solo se viene effettuato il training da zero e non un retraining", choices = ['default', 'custom'])
args = parser.parse_args()

env = gym.make(args.env)

# Questa funzione serve per scrivere su tensorboard
def callback(locals_, globals_):
    self_ = locals_['self']
    if env.done:
        final_impr = (env.initial_sum-env.final_sum)/env.initial_sum*100
        mff_impr =(env.initial_sum-env.mff_sum)/env.initial_sum*100
        over_mff = final_impr - mff_impr
        summary = tf.Summary(value=[tf.Summary.Value(tag='improvement', simple_value=final_impr)])
        summary2 = tf.Summary(value=[tf.Summary.Value(tag='improvement over mff', simple_value=over_mff)])
        locals_['writer'].add_summary(summary, self_.num_timesteps)
        locals_['writer'].add_summary(summary2, self_.num_timesteps)
    return True

# Qui si deve devinire l'architettura della rete se non e' stata scelta quella di default
class CustomDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs,
                                           layers=[32, 32],
                                           layer_norm=False,
                                           feature_extraction="mlp")


path = os.getenv("HOME")+"/models/"+args.model_name
if args.retrain == "no_retrain":
	# Instantiate the agent
	if args.policy == 'custom':
		policy = CustomDQNPolicy
	else:
		policy = "CnnPolicy" if "Img" in args.env else "MlpPolicy"
	model = DQN(policy, env, gamma = 1, learning_rate=args.lr, prioritized_replay=True, verbose=1,tensorboard_log=os.getenv("HOME")+"/tensorboard/"+args.model_name)
	# Train the agent
	model.learn(total_timesteps=int(args.steps),callback=callback)
	# Save the agent
	model.save(path)
else:
	# Load and train the agent
	path = os.getenv("HOME")+"/models/"+args.model_name
	model = DQN.load(path,tensorboard_log=os.getenv("HOME")+"/tensorboard/"+args.model_name+"_retrained")
	model.set_env(env)
	model.learn(total_timesteps=int(args.steps),callback=callback)
	# Save the agent
	path = os.getenv("HOME")+"/models/"+args.model_name+"_retrained"
	model.save(path)
