#!/usr/bin/python3
import numpy as np
import gymnasium as gym
import time
from mc_maze_agent import Agent

'''@brief transform the state value to a row and column in the environment
'''
def get_row_column(state):
    row = int(state/5)
    column = state - (row * 5)
    return row, column


'''@brief simulate the environment with the agents taught policy
'''
def simulate_episodes(agent, num_episodes=3):
    description=['SFFFH', 'FFHFF', 'FFFFF', 'HFFFF', 'FFFFG']
    env2 = gym.make('FrozenLake-v1',  desc=description, render_mode="human", map_name=None, is_slippery=False)
    for _ in range(num_episodes):
        done = False
        state, _ = env2.reset()
        env2.render()
        while not done:
            # Random choice from behavior policy
            action = int(agent.pi[state])
            # take a step
            env2.render()
            time.sleep(0.1)
            next_state, _, done, _, _ = env2.step(action)
            state = next_state
        time.sleep(1.0)

def value_iteration(env, gamma=0.9, theta=1e-10):
    V = np.zeros(env.observation_space.n)
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    while True:
        delta = 0
        for s in range(env.observation_space.n):
            v = V[s]
            Q_s = np.zeros(env.action_space.n)
            for a in range(env.action_space.n):
                for prob, next_state, reward, done in env.P[s][a]:
                    Q_s[a] += prob * (reward + gamma * V[next_state])
            V[s] = max(Q_s)
            Q[s] = Q_s
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    policy = np.argmax(Q, axis=1)
    return Q, policy

def main():
    # MODE = ["LEARN"|"LOAD"]
    MODE = "LEARN"
    NUM_EPISODES = 50000

    # Define the Blackjack environment
    description=['SFFFH', 'FFHFF', 'FFFFF', 'HFFFF', 'FFFFG']
    env = gym.make('FrozenLake-v1', desc=description, map_name=None, is_slippery=False)

    # Define the behavior policy (random policy for exploration)
    behavior_policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
    # Define the target policy (random)
    target_policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n

    # Set hyperparameters
    gamma = 0.9
    action_desc = ["<", "|", ">", "^"]

    # Apply off-policy Monte Carlo algorithm
    agent = Agent(env, behavior_policy, target_policy, gamma)
    if MODE == "LEARN":
        # SAMPLE LEARNED POLICY -- Update after running for longer
        Q_mc, pi_mc = agent.run_MC(NUM_EPISODES)
    elif MODE == "LOAD":
        Q_mc = np.array(
        [[0.531441,   0.59049   , 0.59049,    0.531441  ],
        [0.531441,   0.        , 0.6561 ,    0.59049   ],
        [0.59049 ,   0.72850953, 0.59049,    0.6561    ],
        [0.6561  ,   0.        , 0.59049,    0.59049   ],
        [0.59049 ,   0.6561    , 0.     ,    0.531441  ],
        [0.      ,   0.        , 0.     ,    0.        ],
        [0.      ,   0.81      , 0.     ,    0.6561    ],
        [0.      ,   0.        , 0.     ,    0.        ],
        [0.6561  ,   0.        , 0.729  ,    0.59049   ],
        [0.6561  ,   0.81      , 0.81   ,    0.        ],
        [0.729   ,   0.9       , 0.     ,    0.72855358],
        [0.      ,   0.        , 0.     ,    0.        ],
        [0.      ,   0.        , 0.     ,    0.        ],
        [0.      ,   0.80928255, 0.9    ,    0.729     ],
        [0.81    ,   0.9       , 1.     ,    0.81      ],
        [0.      ,   0.        , 0.     ,    0.        ]])
        agent.Q = Q_mc
        agent.update_greedy_policy()
        pi_mc = agent.pi

    print("Estimated Action-Value Function (MC):")
    print(Q_mc)

    print("\nTarget Policy (MC):")
    pi_mat_mc = np.empty((5, 5), dtype=str)
    for i, action in enumerate(pi_mc):
        row, column = get_row_column(i)
        pi_mat_mc[row][column] = action_desc[action]
    print(pi_mat_mc)

    # Perform Value Iteration
    Q_vi, pi_vi = value_iteration(env, gamma)

    print("\nEstimated Action-Value Function (Value Iteration):")
    print(Q_vi)

    print("\nTarget Policy (Value Iteration):")
    pi_mat_vi = np.empty((5, 5), dtype=str)
    for i, action in enumerate(pi_vi):
        row, column = get_row_column(i)
        pi_mat_vi[row][column] = action_desc[action]
    print(pi_mat_vi)

    simulate_episodes(agent, num_episodes=3)

if __name__ == "__main__":
    main()

