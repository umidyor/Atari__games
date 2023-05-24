import gymnasium as gym
game_name="LunarLander-v2"
env = gym.make(game_name, render_mode="human")

episodes = 10
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)[:4]
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()