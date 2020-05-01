import gym
import gym_qapConst
import numpy as np
import os
import tensorflow as tf
from pathlib import Path
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.noise import AdaptiveParamNoiseSpec
from stable_baselines.common.callbacks import BaseCallback


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, model_name: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = str(Path(log_dir+"/"+model_name))
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last n episodes
              mean_reward = np.mean(y[-50:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, env, check_freq: int, verbose=0):
        self.is_tb_set = False
        self.check_freq = check_freq
        self.env = env
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log additional tensor
        if self.n_calls % self.check_freq == 0:

          # Log scalar value
          final_impr = (self.env.initial_sum-self.env.final_sum)/self.env.initial_sum*100
          summary = tf.Summary(value=[tf.Summary.Value(tag='% improvement wrt initial state', simple_value=final_impr)])
          self.locals['writer'].add_summary(summary, self.num_timesteps)
          mff_impr =(self.env.initial_sum-self.env.mff_sum)/self.env.initial_sum*100
          over_mff = final_impr - mff_impr
          summary = tf.Summary(value=[tf.Summary.Value(tag='% improvement wrt mff', simple_value=over_mff)])
          self.locals['writer'].add_summary(summary, self.num_timesteps)
          return True