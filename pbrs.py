import numpy as np


class PBRS:
    def __init__(self, n_states):
        self.n_states = n_states
        self.discount = 0.1
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

    def set_potentials(self, gw):
        """
        Set the potentials of states based on target locations
        """
        poi_state_potentials = [[] for i in range(self.n_states)]
        for tloc in gw.targets:
            for x in range(gw.width):
                for y in range(gw.height):
                    state = x + gw.height*y
                    x_dist = abs(tloc[0] - x)
                    y_dist = abs(tloc[1] - y)
                    n_moves = x_dist + y_dist  # Number of moves from current state to POI
                    poi_state_potentials[state].append(-n_moves)  # potential is negative number of moves to POI

        for state in range(self.n_states):
            self._initial_potentials[state] = sum(poi_state_potentials[state])/len(poi_state_potentials[state])
        self.state_potentials = self._initial_potentials.copy()

    def update_potentials(self, reward, state):
        """
        Update state potentials
        """
        current_phi = self.state_potentials[state]
        new_phi = current_phi + 0.1*reward
        self.state_potentials[state] = new_phi

