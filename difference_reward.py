import numpy as np


def calc_difference_reward(g_reward, gw):
    """
    Calculate the difference reward for each agent
    """
    difference_reward = np.zeros(len(gw.agents))

    # Count number of agents at a target
    for i in range(len(gw.agents)):
        target_capture_counter = np.zeros(len(gw.targets))
        for t_id, t_loc in enumerate(gw.targets):
            for a_id, ag in enumerate(gw.agents):
                if t_loc == gw.agents[ag].loc and a_id != i:  # Counterfactual that ignores contributions of agent i
                    target_capture_counter[t_id] += 1

        # Count how many targets are captured in counterfactual state
        target_count = 0
        for agent_count in target_capture_counter:
            if agent_count > 0:
                target_count += 1

        counterfactual_global_reward = 0
        if target_count > 0:
            counterfactual_global_reward = target_count/len(gw.targets)
        difference_reward[i] = g_reward - counterfactual_global_reward

    return difference_reward
