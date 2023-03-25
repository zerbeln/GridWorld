from agent import Agent, QLearner
import random


class GridWorld:
    def __init__(self, width, height):
        self.width = width
        self.height = height
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

        a_loc = []
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
        Check collision with world boundaries
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

        # Return reward and new agent state
        if [x, y] in self.targets:
            return self.reward, [x, y]
        else:
            return 0, [x, y]
