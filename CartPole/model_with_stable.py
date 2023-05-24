# from stable_baselines3 import PPO
# import gym
#
# env = gym.make("CartPole-v1")
#
# model = PPO(policy = "MlpPolicy",env =  env, verbose=1)
# model.learn(total_timesteps=25000)
#
# obs = env.reset()
# for i in range(1000):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#       obs = env.reset()


"""Thank you for your attention:)"""


# from stable_baselines3 import PPO
# import gym
#
# env = gym.make("CartPole-v1")
#
# model = PPO(policy="MlpPolicy", env=env, verbose=1)
# model.learn(total_timesteps=25000)
#
# episodes = 20  # Number of episodes to run
# for episode in range(episodes):
#     obs = env.reset()
#     done = False
#     total_reward = 0
#
#     while not done:
#         action, _state = model.predict(obs, deterministic=True)
#         obs, reward, done, info = env.step(action)
#         env.render()
#         total_reward += reward
#
#     print(f"Episode {episode + 1}: Total Reward = {total_reward}")
#
# env.close()


from stable_baselines3 import PPO
import gym

env = gym.make("CartPole-v1")

model = PPO(policy="MlpPolicy", env=env, verbose=1)
model.learn(total_timesteps=25000)

episodes = 20
for episode in range(1, episodes + 1):
    obs = env.reset()
    score = 0

    while True:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        score += reward
        env.render()

        if done:
            print('Episode: {} Score: {}'.format(episode, score))
            break

env.close()