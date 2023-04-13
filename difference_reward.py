import numpy as np


def calc_difference_reward(g_reward, gw):
    """
    Calculate the difference reward for each agent
    """
    difference_reward = np.zeros(len(gw.agents))

    # Count number of agents at a target
    for i in range(len(gw.agents)):
        counterfactual_global_reward = 0
        target_capture_counter = np.zeros(len(gw.targets))
        for id, loc in enumerate(gw.targets):
            for ag in range(len(gw.agents)):
                if loc == gw.agents[f'A{ag}'].loc and ag != i:  # Counterfactual that ignores contributions of agent i
                    target_capture_counter[id] += 1

        # If target has at least one agent, targets increases reward
        for tcount in target_capture_counter:
            if tcount > 0:
                counterfactual_global_reward += gw.reward

        difference_reward[i] = counterfactual_global_reward - g_reward

    return difference_reward
