from stable_baselines import DQN
from stable_baselines.gail import generate_expert_traj
import gym
import gym_qapConst
import tensorflow as tf
import os
import argparse
from stable_baselines.deepq.policies import FeedForwardPolicy
from pathlib import Path
from Callbacks import SaveOnBestTrainingRewardCallback
from Callbacks import TensorboardCallback
from stable_baselines.bench import Monitor

parser = argparse.ArgumentParser()
parser.add_argument("num_prod", type=int, help = "Numero prodotti (deve essere un quadrato)")
parser.add_argument("steps", type=float, help = "Numero di steps da effettuare Es.[2e5]")
parser.add_argument("lr", type=float, help = "learning_rate Es[1e-4]")
parser.add_argument("model_name", type=str, help = "nome del modello da caricare")
parser.add_argument("env", type=str, help = "Nome dell'environment: Const lo stato iniziale e' fisso, \
    Img l'osservazione e' basata sull'immagine ", choices = ['qapConst-v0', 'qapImgConst-v0', 'qap-v0', 'qapImg-v0'])
parser.add_argument("max_swaps", type=int, help = "Numero massimo di swap consentiti prima di terminare il game")
args = parser.parse_args()

Path(os.getenv("HOME")+"/models/"+args.model_name).mkdir(parents=True, exist_ok=True)
path = str(Path(os.getenv("HOME")+"/models/"+args.model_name))
env = gym.make(args.env)
env = Monitor(env, path)

#Definizioni delle callback da usare
callbackSaveBestModel = SaveOnBestTrainingRewardCallback(args.max_swaps, path, args.model_name+"_retrained", verbose=0)
callbackTensorboard = TensorboardCallback(env,args.max_swaps, verbose=1)

model = DQN.load(path+"/"+args.model_name,tensorboard_log=Path(os.getenv("HOME")+"/tensorboard/"+args.model_name+"_retrained"))
model.set_env(env)
model.learn(total_timesteps=int(args.steps),callback=[callbackSaveBestModel,callbackTensorboard])