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
        self.target_proximity_potential(gw, n_steps)

    def exploration_potential(self, gw, agent_loc, n_steps):
        """
        Potential function that incentivizes agents travelling away from their starting point
        """
        for x in range(gw.width):
            for y in range(gw.height):
                state = x + gw.height * y
                x_dist = abs(x - agent_loc[0])
                y_dist = abs(y - agent_loc[1])
                self._initial_potentials[state] = n_steps - (x_dist + y_dist)

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
                    poi_state_potentials[state].append(n_steps - (x_dist + y_dist))

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

