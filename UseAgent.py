import gym
import gym_qap
import gym_qapImg
import matplotlib.pyplot as plt


from stable_baselines import DQN
from stable_baselines import A2C
from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy



env = gym.make('qap-v0')

# Load the trained agent
model = DQN.load("./models/test",tensorboard_log="./tensorboard/")
model.set_env(env)

# Evaluate the agent
mean_reward, n_steps = evaluate_policy(model, env, n_eval_episodes=10)


# Enjoy trained agent
num_simulations = 500
x = [i for i in range(0,num_simulations)] 
y1 = [0 for j in range(0,num_simulations)]

for j in range(num_simulations):
	obs = env.reset()
	for i in range(60):
		action, _states = model.predict(obs)
		obs, rewards, dones, info = env.step(action)
	#env.render()
	y1[j] = ((env.initial_sum-env.current_sum)/env.initial_sum*100)-((env.initial_sum-env.mff_sum)/env.initial_sum*100)
# x axis values 
# corresponding y axis values 
  
# plotting the points  
plt.plot(x, y1, label = "Agent improvement over mff") 

  
# naming the x axis 
plt.xlabel('Simulation') 
# naming the y axis 
plt.ylabel('% improvement ') 
  
# giving a title to my graph 
plt.title('Imrovement Graph') 
  
# function to show the plot 
plt.show() 
