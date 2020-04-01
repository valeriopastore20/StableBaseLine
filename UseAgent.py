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
from stable_baselines.common.evaluation import evaluate_policy



parser = argparse.ArgumentParser()
parser.add_argument("num_prod", type=int, help="Numero di prodotti su cui il modello e' stato allenato")
parser.add_argument("num_sim", type=int, help="Numero di simulazioni da effettuare")
parser.add_argument("model_name", type=str, help="Nome del modello da testare")
parser.add_argument("env", type=str, help="Nome dell'environment: Const lo stato iniziale e' fisso, \
    Img l'osservazione e' basata sull'immagine ", choices = ['qapConst-v0', 'qapImgConst-v0', 'qap-v0', 'qapImg-v0'])
args = parser.parse_args()

env = gym.make(args.env)


# Load the trained agent
path = os.getenv("HOME")+"/models/"+args.model_name
model = DQN.load(path)
model.set_env(env)

# Evaluate the agent
#mean_reward, n_steps = evaluate_policy(model, env, n_eval_episodes=10)


# Enjoy trained agent
num_simulations = args.num_sim
x = [i for i in range(0,num_simulations)] 
y1 = [0 for j in range(0,num_simulations)]
y2 = [0 for j in range(0,num_simulations)]

for j in range(num_simulations):
	obs = env.reset()
	while not env.done:
		action, _states = model.predict(obs)
		obs, rewards, dones, info = env.step(action)
	#env.render()
#	y1[j] = ((env.initial_sum-env.mff_sum)/env.initial_sum*100)
	y2[j] = ((env.initial_sum-env.final_sum)/env.initial_sum*100)-((env.initial_sum-env.mff_sum)/env.initial_sum*100)
# x axis values 
# corresponding y axis values 
  
# plotting the points  
plt.plot(x,y2, label = "Agent improvement over mff") 

  
# naming the x axis 
plt.xlabel('Simulation') 
# naming the y axis 
plt.ylabel('% improvement ') 
  
# giving a title to my graph 
plt.title('Imrovement Graph') 
  
# function to show the plot 
plt.show() 
