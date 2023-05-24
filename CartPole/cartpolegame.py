import gym
import random


def game():
    env = gym.make('CartPole-v1')
    states = env.observation_space.shape[0]
    actions = env.action_space.n
    episodes = 20
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        score = 0
        # env.render()
        while not done:
            env.render()
            action = random.choice([0, 1])
            step_result = env.step(action)
            next_state, reward, done, info = step_result[:4]
            score += reward

        print('Episode:{} Score:{}'.format(episode, score))

    env.close()


if __name__ == "__main__":
    game()