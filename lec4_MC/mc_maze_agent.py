#!/usr/bin/python3
import numpy as np

class Agent:
    def __init__(self, env, behavior_policy, target_policy, gamma ) -> None:
        self.env = env
        self.behavior_policy = behavior_policy
        self.target_policy = target_policy
        self.gamma = gamma
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))
        self.C = np.zeros((env.observation_space.n, env.action_space.n))
        self.pi = np.zeros(env.observation_space.n, dtype=int)

    def generate_episode(self):
        episode = []
        state, _ = self.env.reset()
        done = False
        while not done:
            action = np.random.choice(self.env.action_space.n, p=self.behavior_policy[state])
            next_state, reward, done, _, _ = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
        return episode

    def update_greedy_policy(self):
        for s in range(self.env.observation_space.n):
            self.pi[s] = np.argmax(self.Q[s])

    def run_MC(self, num_episodes):
        for episode_idx in range(num_episodes):
            episode = self.generate_episode()
            G = 0
            W = 1
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = self.gamma * G + reward
                self.C[state][action] += W
                self.Q[state][action] += (W / self.C[state][action]) * (G - self.Q[state][action])
                W *= self.target_policy[state][action] / self.behavior_policy[state][action]
                if W == 0:
                    break
            self.update_greedy_policy()
        return self.Q, self.pi

    '''@brief Retrieve the action value of the current (state, action) pair
        @note These methods can be changed for different matrix shapes without changing the core code
    '''
    def get_action_value(self, state, action):
        return self.Q[state, action]

    def get_cum_sum_weights(self, state, action):
        return self.C[state, action]

    def update_action_value(self, state, action, q):
        self.Q[state, action] = q

    def update_cum_sum_weights(self, state, action, c):
        self.C[state, action] = c

    def query_target_policy_action(self, state, action):
        return self.target_policy[state, action]

    def query_behavior_policy_action(self, state, action):
        return self.behavior_policy[state, action]
