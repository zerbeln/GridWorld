import random
import numpy as np


class Agent:
    def __init__(self, x, y):
        self.loc = [x, y]
        self.initial_position = (x, y)
        self.actions = [0, 1, 2, 3]  # Up, Down, Left, Right

    def reset_agent(self):
        """
        Reset the agent to its initial position
        """
        self.loc[0] = self.initial_position[0]
        self.loc[1] = self.initial_position[1]


class QLearner(Agent):
    def __init__(self, n_states, x, y):
        super().__init__(x, y)
        self.n_states = n_states
        self.n_actions = len(self.actions)
        self.discount = 0.9
        self.alpha = 0.1
        self.epsilon = 0.1
        self.q_table = np.zeros((n_states, len(self.actions)))

    def update_q_val(self, state, action, next_state, reward):
        """
        Update q-values after a state transition
        """
        q_val = self.q_table[state, action]
        max_q = np.max(self.q_table[next_state])

        new_q = q_val + self.alpha*(reward + self.discount*max_q - q_val)
        self.q_table[state, action] = new_q

    def get_egreedy_action(self, state):
        """
        Choose an action with e-greedy selection
        """
        rand_val = random.uniform(0, 1)
        if rand_val < self.epsilon:
            return np.argmax(self.q_table[state])
        else:
            return random.randint(0, self.n_actions-1)

    def get_greedy_action(self, state):
        """
        Only choose the action with the highest value estimate
        """
        return np.argmax(self.q_table[state])

    def reset_learner(self):
        """
        Clear data in the q-table
        """
        self.q_table = np.zeros((self.n_states, self.n_actions))
