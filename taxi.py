import gym
import numpy as np

class QLAgent:

    def __init__(self, action_size, state_size):
        self.epsilon = 0.2
        self.action_size = action_size
        self.state_size = state_size
        self.learning_rate = 0.90
        self.gamma = 0.10
        self.q_table = np.zeros([self.state_size, self.action_size])
        self.reward = 0

    def act(self, state, play=False):
        if np.random.random_sample() <= 0.2 and not play:
            return np.random.randint(0, self.action_size)
        else:
            return np.argmax(self.q_table[state, :])

    def predict(self, state):
        return np.argmax(self.q_table[state, :])

    def update_q_table(self, state, action, reward, next_state):
        self.reward += reward
        self.q_table[state, action] = self.q_table[state, action] +\
            self.learning_rate * (reward + self.gamma*np.max(self.q_table[next_state, :]) - self.q_table[state, action])

    def save_model(self, file_name):
        np.save(file_name, self.q_table)

    def load_model(self, file_name):
        self.q_table = np.load(file_name)

def train():
    env = gym.make('Taxi-v2')
    dqlagent = QLAgent(env.action_space.n, env.observation_space.n)
    for i in range(1000):
        state = env.reset()
        while True:
            action = dqlagent.act(state)
            next_state, rew, done, info = env.step(action)
            dqlagent.update_q_table(state, action, rew, next_state)
            state = next_state
            if done:
                print(dqlagent.reward)
                break

    dqlagent.save_model('modelo.npy')

def play():
    env = gym.make('Taxi-v2')
    dqlagent = QLAgent(env.action_space.n, env.observation_space.n)
    dqlagent.load_model('save_model/modelo.npy')
    state = env.reset()
    while True:
        action = dqlagent.act(state, True)
        next_state, rew, done, _ = env.step(action)
        env.render()
        input()
        state = next_state
        if done:
            return (dqlagent.reward)

import sys
if __name__ == '__main__':
    if 'train' in sys.argv:
        train()
    elif 'play' in sys.argv:
        play()
