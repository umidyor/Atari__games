from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import gym
def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1, states)))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    states = env.observation_space.shape[0]
    actions = env.action_space.n
    model = build_model(states, actions)
    print(model.summary())