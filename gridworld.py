from agent import QLearner
import random
import numpy as np
import os
import csv


class GridWorld:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.n_states = height * width
        self.reward = 10
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

        self.save_configuration()

    def save_configuration(self):
        """
        Save the Gridworld configuration to a CSV file
        """
        dir_name = 'World_Config'  # Intended directory for output files

        if not os.path.exists(dir_name):  # If Data directory does not exist, create it
            os.makedirs(dir_name)

        tfile_name = os.path.join(dir_name, f'Target_Config.csv')
        afile_name = os.path.join(dir_name, f'Agent_Config.csv')

        with open(tfile_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for t_loc in self.targets:
                writer.writerow(t_loc)

        with open(afile_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for ag in self.agents:
                writer.writerow(self.agents[ag].loc)

        csvfile.close()

    def load_configuration(self, n_agents, n_targets):
        """
        Load Gridworld configuration from CSV files
        """
        csv_target_input = []
        csv_agent_input = []
        with open(f'World_Config/Target_Config.csv') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')

            for row in csv_reader:
                csv_target_input.append(row)

        for target_id in range(n_targets):
            tx = float(csv_target_input[target_id][0])
            ty = float(csv_target_input[target_id][1])

            self.targets.append([tx, ty])

        with open(f'World_Config/Agent_Config.csv') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')

            for row in csv_reader:
                csv_agent_input.append(row)

        for agent_id in range(n_agents):
            ax = float(csv_agent_input[agent_id][0])
            ay = float(csv_agent_input[agent_id][1])

            self.agents[f'A{agent_id}'] = QLearner(self.width*self.height, int(ax), int(ay))

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
            return -1, [x, y]

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
            else:
                global_reward -= 1

        return global_reward
