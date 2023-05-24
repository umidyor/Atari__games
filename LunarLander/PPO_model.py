import numpy as np
import gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
env = gym.make("LunarLander-v2")
env = DummyVecEnv([lambda: env])
# model = PPO('MlpPolicy', env, verbose = 1)
# model.learn(total_timesteps=100000)
# evaluate_policy(model, env, n_eval_episodes=10, render=True)
# env.close()
# model.save("PPO_model")

model = PPO.load("PPO_model.zip", env=env)
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
env.close()