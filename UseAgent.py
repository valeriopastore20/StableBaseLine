import gym
import gym_qap
import gym_qapConst
import gym_qapImg
import gym_qapImgConst
import matplotlib.pyplot as plt
import os
import argparse

from stable_baselines import DQN
from stable_baselines import A2C
from stable_baselines import PPO2
from pathlib import Path


"""
	Questo programma consente di testare un agente gia' allenato e crea
	un grafico dei risultati ottenuti nelle varie simulazioni rispetto
	al risultato ottenuto adottando la disposizione mff
"""

parser = argparse.ArgumentParser()
parser.add_argument("num_prod", type=int, help="Numero di prodotti su cui il modello e' stato allenato")
parser.add_argument("num_sim", type=int, help="Numero di simulazioni da effettuare")
parser.add_argument("model_name", type=str, help="Nome del modello da testare")
parser.add_argument("env", type=str, help="Nome dell'environment: Const lo stato iniziale e' fisso, \
    Img l'osservazione e' basata sull'immagine ", choices = ['qapConst-v0', 'qapImgConst-v0', 'qap-v0', 'qapImg-v0'])
parser.add_argument("max_swaps", type=int, help = "Numero massimo di swap consentiti prima di terminare il game")

args = parser.parse_args()
env = gym.make(args.env)


# Load the trained agent
path = Path(os.getenv("HOME")+"/models/"+args.model_name)
model = DQN.load(path)
model.set_env(env)

# Enjoy trained agent
num_simulations = args.num_sim
x = [i for i in range(0,num_simulations)] 
y = [0 for j in range(0,num_simulations)]

for j in range(num_simulations):
	obs = env.reset()
	print("Disposizione iniziale: ")
	print(env.matrix_pl)
	while not env.done:
		action, _states = model.predict(obs)
		obs, rewards, dones, info = env.step(action)
		print(env.action)
	env.render()
	print("Disposizione finale: ")
	print(env.matrix_pl)
	y[j] = ((env.initial_sum-env.final_sum)/env.initial_sum*100)-((env.initial_sum-env.mff_sum)/env.initial_sum*100)

# plotting the points  
plt.plot(x,y, label = "Agent improvement over mff") 

# naming the x axis 
plt.xlabel('Simulation') 
# naming the y axis 
plt.ylabel('% improvement ') 
  
# giving a title to my graph 
plt.title('Imrovement Graph') 
  
# function to show the plot 
plt.show() 
