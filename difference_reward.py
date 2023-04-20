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
        for t_id, loc in enumerate(gw.targets):
            for a_id, ag in enumerate(gw.agents):
                if loc == gw.agents[ag].loc and a_id != i:  # Counterfactual that ignores contributions of agent i
                    target_capture_counter[t_id] += 1

        # If target has at least one agent, targets increases reward
        for tcount in target_capture_counter:
            if tcount > 0:
                counterfactual_global_reward += gw.reward

        difference_reward[i] = g_reward - counterfactual_global_reward

    return difference_reward
