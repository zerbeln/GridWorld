import numpy as np


class PBRS:
    def __init__(self, n_states):
        self.n_states = n_states
        self.discount = 0.9
        self.state_potentials = np.zeros(n_states)
        self._initial_potentials = np.zeros(n_states)  # Used for resetting potentials to original seed

    def potential_function(self, current_state, prev_state):
        """
        Calculates change in potential over a state transition
        """
        phi_s1 = self.state_potentials[prev_state]
        phi_s2 = self.discount*self.state_potentials[current_state]
        delta_potential = phi_s2 - phi_s1

        return delta_potential

    def reset_potentials(self):
        """
        Reset the state potential estimates
        """
        self.state_potentials = self._initial_potentials.copy()

    def random_init(self):
        """
        Create a random initialization for state potentials
        """
        self._initial_potentials = np.random.normal(0, 1, self.n_states)
        self.state_potentials = self._initial_potentials.copy()

    def set_potentials(self, gw, agent_id, n_steps):
        """
        Set the potentials of states based on target locations
        """
        # self.exploration_potential(gw, gw.agents[f'A{agent_id}'].initial_position, n_steps)
        # self.target_proximity_potential(gw, n_steps)
        self.poi_agent_distance(gw, n_steps, agent_id)

    def exploration_potential(self, gw, agent_loc, n_steps):
        """
        Potential function that incentivizes agents travelling away from their starting point
        """
        for x in range(gw.width):
            for y in range(gw.height):
                state = x + gw.height * y
                x_dist = abs(x - agent_loc[0])
                y_dist = abs(y - agent_loc[1])
                if [x, y] in gw.targets:
                    self._initial_potentials[state] = 1.0
                else:
                    self._initial_potentials[state] = (x_dist + y_dist) / n_steps

        self.state_potentials = self._initial_potentials.copy()

    def target_proximity_potential(self, gw, n_steps):
        """
        Potential function that incentivizes agent travelling towards POI
        """
        poi_state_potentials = [[] for i in range(self.n_states)]
        for t_loc in gw.targets:
            for x in range(gw.width):
                for y in range(gw.height):
                    state = x + gw.height * y
                    x_dist = abs(x - t_loc[0])
                    y_dist = abs(y - t_loc[1])
                    poi_state_distance = x_dist + y_dist
                    if [x, y] in gw.targets:
                        poi_state_potentials[state].append(1.0)
                    else:
                        potential = 1 - (poi_state_distance/n_steps)
                        if potential < 0:
                            potential = 0
                        poi_state_potentials[state].append(potential)

        for state in range(self.n_states):
            self._initial_potentials[state] = np.mean(poi_state_potentials[state])
        self.state_potentials = self._initial_potentials.copy()

    def poi_agent_distance(self, gw, n_steps, agent_id):
        """
        Potential function that incentivizes agent travelling towards closer POI
        """
        poi_state_potentials = [[] for i in range(self.n_states)]
        agent_loc = gw.agents[f'A{agent_id}'].loc
        for t_loc in gw.targets:
            for x in range(gw.width):
                for y in range(gw.height):
                    state = x + gw.height * y
                    poi_state_dist = abs(x - t_loc[0]) + abs(y - t_loc[1])
                    state_agent_dist = abs(agent_loc[0] - x) + abs(agent_loc[1] - y)
                    poi_agent_dist = abs(agent_loc[0] - t_loc[0]) + abs(agent_loc[1] - t_loc[1])

                    # use max
                    if (poi_state_dist + state_agent_dist) == poi_agent_dist:
                        potential = 1 - (poi_state_dist/(poi_state_dist + state_agent_dist))
                    else:
                        potential = 0.0
                    # if 0 < poi_state_dist:
                    #     potential = (state_agent_dist/poi_state_dist)/(gw.height + gw.width - 2)  # use sum
                    # else:
                    #     potential = 1.0
                    poi_state_potentials[state].append(potential)

        for state in range(self.n_states):
            self._initial_potentials[state] = max(poi_state_potentials[state])
        self.state_potentials = self._initial_potentials.copy()

    def update_potentials(self, reward, state):
        """
        Update state potentials
        """
        current_phi = self.state_potentials[state]
        new_phi = current_phi + 0.1*reward
        self.state_potentials[state] = new_phi

