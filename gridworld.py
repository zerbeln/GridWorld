from agent import QLearner
import random
import numpy as np


class GridWorld:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.n_states = height * width
        self.reward = 100
        self.agents = {}  # Dictionary for agent objects
        self.targets = []  # Coordinates of targets in the Gridworld
        self.walls = []  # Coordinates of cells that are walls in the Gridworld

    def create_world(self, n_agents, n_targets):
        for t in range(n_targets):
            x = random.randint(0, self.width-1)
            y = random.randint(0, self.height-1)

            # Make sure targets are not placed on top of each other
            while [x, y] in self.targets:
                x = random.randint(0, self.width-1)
                y = random.randint(0, self.height-1)

            self.targets.append([x, y])

        a_loc = []  # Agent locations
        for a in range(n_agents):
            x = random.randint(0, self.width-1)
            y = random.randint(0, self.height-1)

            # Make sure agents do not start out on top of other agents or targets
            while [x, y] in self.targets or [x, y] in a_loc:
                x = random.randint(0, self.width-1)
                y = random.randint(0, self.height-1)

            a_loc.append([x, y])
            self.agents[f'A{a}'] = QLearner(self.width*self.height, x, y)

    def check_collision(self, x, y):
        """
        Check for collision with world boundaries
        """
        if x < 0 or x >= self.width:
            return True
        elif y < 0 or y >= self.height:
            return True
        else:
            return False

    def step(self, agent_state, action):
        """
        Agent provides an action, step function provides state-transition and reward
        """
        x, y = agent_state

        # Provide state-transition for agent
        if action == 0:  # Up
            collision = self.check_collision(x, y+1)
            if not collision:
                y += 1
        elif action == 1:  # Down
            collision = self.check_collision(x, y-1)
            if not collision:
                y -= 1
        elif action == 2:  # Left
            collision = self.check_collision(x-1, y)
            if not collision:
                x -= 1
        elif action == 3:  # Right
            collision = self.check_collision(x+1, y)
            if not collision:
                x += 1
        else:
            assert(action == 4)  # Else the agent remains stationary

        # Return local agent reward and new agent state
        if [x, y] in self.targets:
            return self.reward, [x, y]
        else:
            return 0, [x, y]

    def calculate_g_reward(self):
        """
        Calculate the global reward for the team of agents
        """
        global_reward = 0

        # Count number of agents at a target
        target_capture_counter = np.zeros(len(self.targets))
        for id, loc in enumerate(self.targets):
            for ag in self.agents:
                if loc == self.agents[ag].loc:
                    target_capture_counter[id] += 1

        # If target has at least one agent, targets increases reward
        for tcount in target_capture_counter:
            if tcount > 0:
                global_reward += self.reward

        return global_reward
