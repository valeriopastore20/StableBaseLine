import gym
import gym_qapConst
import tensorflow as tf
import os
import argparse
from stable_baselines.deepq.policies import FeedForwardPolicy
from stable_baselines import DQN
from pathlib import Path
from Callbacks import SaveOnBestTrainingRewardCallback
from Callbacks import TensorboardCallback
from stable_baselines.bench import Monitor


parser = argparse.ArgumentParser()
parser.add_argument("num_prod", type=int, help = "Numero prodotti (deve essere un quadrato)")
parser.add_argument("steps", type=float, help = "Numero di steps da effettuare Es.[2e5]")
parser.add_argument("lr", type=float, help = "learning_rate Es[1e-4]")
parser.add_argument("model_name", type=str, help = "nome del modello da salvare")
parser.add_argument("env", type=str, help = "Nome dell'environment: Const lo stato iniziale e' fisso, \
    Img l'osservazione e' basata sull'immagine ", choices = ['qapConst-v0', 'qapImgConst-v0', 'qap-v0', 'qapImg-v0'])
parser.add_argument("policy", type=str, help = "Specifica se deve essere usata un'architettur di default oppure \
	una custom (Ovviamente ha effetto solo se viene effettuato il training da zero e non un retraining", choices = ['default', 'custom'])
parser.add_argument("max_swaps", type=int, help = "Numero massimo di swap consentiti prima di terminare il game")

args = parser.parse_args()

env = gym.make(args.env)
Path(os.getenv("HOME")+"/models/"+args.model_name).mkdir(parents=True, exist_ok=True)
path = str(Path(os.getenv("HOME")+"/models/"+args.model_name))
env = Monitor(env, path)

# Qui si deve definire l'architettura della rete se non e' stata scelta quella di default
class CustomDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs,
                                           layers=[128, 128],
                                           layer_norm=True,
                                           feature_extraction="mlp")

#Definizioni delle callback da usare
callbackSaveBestModel = SaveOnBestTrainingRewardCallback(args.max_swaps, path, args.model_name, verbose=0)
callbackTensorboard = TensorboardCallback(env,args.max_swaps, verbose=1)

#Impostazione policy
if args.policy == 'custom':
	policy = CustomDQNPolicy
else:
	policy = "MlpPolicy"

# Train the agent	
model = DQN(policy, env, learning_rate=args.lr, prioritized_replay=True, double_q=True, verbose=1,tensorboard_log=Path(os.getenv("HOME")+"/tensorboard/"+args.model_name))
model.learn(total_timesteps=int(args.steps),callback=[callbackSaveBestModel,callbackTensorboard])
