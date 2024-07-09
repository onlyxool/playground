#!/usr/bin/python3
import time
import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

#---------------------------
# Helper functions
#---------------------------

'''@brief Describes the environment actions, observation states, and reward range
'''
def describe_env(env: gym.Env):
    num_actions = env.action_space.n
    obs = env.observation_space
    num_obs = env.observation_space.n
    reward_range = env.reward_range
    action_desc = {
        0: "Move south (down)",
        1: "Move north (up)",
        2: "Move east (right)",
        3: "Move west (left)",
        4: "Pickup passenger",
        5: "Drop off passenger"
    }
    print("Observation space: ", obs)
    print("Observation space size: ", num_obs)
    print("Reward Range: ", reward_range)

    print("Number of actions: ", num_actions)
    print("Action description: ", action_desc)
    return num_obs, num_actions


'''@brief Get the string description of the action
'''
def get_action_description(action):
    action_desc = {
        0: "Move south (down)",
        1: "Move north (up)",
        2: "Move east (right)",
        3: "Move west (left)",
        4: "Pickup passenger",
        5: "Drop off passenger"
    }
    return action_desc[action]

'''@brief print full description of current observation
'''
def describe_obs(obs):
    obs_desc = {
        0: "Red",
        1: "Green",
        2: "Yellow",
        3: "Blue",
        4: "In taxi"
    }
    obs_dict = breakdown_obs(obs)
    print("Passenger is at: {0}, wants to go to {1}. Taxi currently at ({2}, {3})".format(
        obs_desc[obs_dict["passenger_location"]],
        obs_desc[obs_dict["destination"]],
        obs_dict["taxi_row"],
        obs_dict["taxi_col"]))

'''@brief Takes an observation for the 'taxi-v3' environment and returns details observation space description
    @details returns a dict with "destination", "passenger_location", "taxi_col", "taxi_row"
    @see: https://gymnasium.farama.org/environments/toy_text/taxi/
'''
def breakdown_obs(obs):
    # ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination = X
    # X % 4 --> destination
    destination = obs % 4
    # X -= remainder, X /= 4
    obs -= destination
    obs /= 4
    # X % 5 --> passenger_location
    passenger_location = obs % 5
    # X -= remainder, X /= 5
    obs -= passenger_location
    obs /= 5
    # X % 5 --> taxi_col
    taxi_col = obs % 5
    # X -= remainder, X /=5
    obs -= taxi_col
    # X --> taxi_row
    taxi_row = obs
    observation_dict= {
        "destination": destination,
        "passenger_location": passenger_location,
        "taxi_row": taxi_row,
        "taxi_col": taxi_col
    }
    return observation_dict


'''@brief simulate the environment with the agents taught policy
'''
def simulate_episodes(env, agent, num_episodes=3):
    for _ in range(num_episodes):
        done = False
        state, _ = env.reset()
        describe_obs(state)
        env.render()
        while not done:
            # Random choice from behavior policy
            action = agent.select_action(state)
            # take a step
            env.render()
            time.sleep(0.1)
            next_state, _, done, _, _ = env.step(action)
            state = next_state
        time.sleep(1.0)


# Hyperparameters
alpha_values = [0.01, 0.001, 0.2]
epsilon_values = [0.2, 0.3]
discount_factor = 0.9
num_episodes = 5000
max_steps = 100

env = gym.make('Taxi-v3')
num_states, num_actions = describe_env(env)


class QLearningAgent:
    def __init__(self, num_states, num_actions, learning_rate, exploration_factor, discount_factor):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.exploration_factor = exploration_factor
        self.discount_factor = discount_factor
        self.q_table = np.zeros((num_states, num_actions))

    def select_action(self, state):
        if random.uniform(0, 1) < self.exploration_factor:
            return random.choice(range(self.num_actions))
        else:
            return np.argmax(self.q_table[state])

    def update_q_value(self, state, action, reward, next_state):
        best_next_action = np.max(self.q_table[next_state])
        self.q_table[state, action] = self.q_table[state, action] + self.learning_rate * (
            reward + self.discount_factor * best_next_action - self.q_table[state, action]
        )


def q_learning(env, learning_rate, exploration_factor):
    agent = QLearningAgent(num_states, num_actions, learning_rate, exploration_factor, discount_factor)
    total_steps_per_episode = []
    return_per_episode = []
    cumulative_reward_per_episode = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward

            agent.update_q_value(state, action, reward, next_state)
            state = next_state
            steps += 1

            if done:
                break

        total_steps_per_episode.append(steps)
        return_per_episode.append(total_reward)
        cumulative_reward_per_episode.append(np.sum(return_per_episode))

    return agent, total_steps_per_episode, return_per_episode, cumulative_reward_per_episode


def plot_metrics(episodes, steps, rewards, alpha, epsilon, filename, window=100):
    avg_cumulative_rewards = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(episodes, steps)
    plt.xlabel('Episode')
    plt.ylabel('Steps per Episode')
    plt.title('Steps per Episode Over Time. Alpha:'+str(alpha)+' Epsilon'+str(epsilon))

    plt.subplot(2, 2, 2)
    plt.plot(episodes, rewards)
    plt.xlabel('Episode')
    plt.ylabel('Return per Episode')
    plt.title('Return per Episode Over Time. Alpha:'+str(alpha)+' Epsilon'+str(epsilon))

    plt.subplot(2, 2, 3)
    plt.plot(episodes, avg_cumulative_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Average Cumulative Reward')
    plt.title('Average Cumulative Reward Over Time. Alpha:'+str(alpha)+' Epsilon'+str(epsilon))

    # Smoothing average cumulative reward
    smoothed_avg_cumulative_rewards = np.convolve(avg_cumulative_rewards, np.ones(window)/window, mode='valid')
    plt.subplot(2, 2, 4)
    plt.plot(episodes[:len(smoothed_avg_cumulative_rewards)], smoothed_avg_cumulative_rewards)
    plt.xlabel('Episode')
    plt.ylabel(f'Avg. Cumulative Reward (Smoothed over {window} episodes)')
    plt.title(f'Smoothed Average Cumulative Reward Over Time. Alpha:'+str(alpha)+' Epsilon'+str(epsilon))
    plt.tight_layout()

    plt.savefig(filename)


def main():
    # Note: Use v3 for the latest version
    env = gym.make('Taxi-v3')
    num_obs, num_actions = describe_env(env)


    # TODO: Train
    # agent = Agent(num_obs, num_actions)
    # agent.train(env, 5000)
    for learning_rate in alpha_values:
        for exploration_factor in epsilon_values:
            agent, steps, returns, cumulative_rewards = q_learning(env, learning_rate, exploration_factor)

            print(f'Learning Rate: {learning_rate}, Exploration Factor: {exploration_factor}')

            # Plot the metrics
            episodes = list(range(1, num_episodes + 1))
            plot_metrics(episodes, steps, returns, learning_rate, exploration_factor, \
                    'metrics_'+str(learning_rate)+'_'+str(exploration_factor)+'.png')


    # TODO: Simulate
    # Note how here, we change the render mode for testing/simulation
    best_learning_rate = 0.2
    best_exploration_factor = 0.2
    agent, steps, returns, cumulative_rewards = q_learning(env, best_learning_rate, best_exploration_factor)

    print(f'Best Learning Rate: {best_learning_rate}, Best Exploration Factor: {best_exploration_factor}')

    env2 = gym.make('Taxi-v3', render_mode="human")
    # simulate_episodes(env2, agent)
    simulate_episodes(env2, agent)


if __name__=="__main__":
    main()
