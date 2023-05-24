import tensorflow as tf
import numpy as np
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from model import build_model
import gym
def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model, memory=memory, policy=policy, nb_actions=actions, nb_steps_warmup=10,target_model_update=1e-2)
    return dqn



env = gym.make('CartPole-v1')
states = env.observation_space.shape[0]
actions = env.action_space.n
model = build_model(states, actions)
dqn = build_agent(model, actions)
dqn.compile(tf.keras.optimizers.legacy.Adam(learning_rate=1e-3), metrics=['mae'])
# dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

# scores=dqn.test(env,nb_episodes=100,visualize=False)
# print(np.mean(scores.history['episode_reward']))

d = dqn.test(env,nb_episodes=20,visualize=True)
print(d)