import numpy as np


def create_counterfactuals(gw):
    """
    Create counterfactuals that agents will use for CFL learning
    """
    counterfactuals = [[0 for i in range(len(gw.targets))] for j in range(len(gw.agents))]
    for a_id, ag in enumerate(gw.agents):
        for t_id, t_loc in enumerate(gw.targets):
            x_dist = abs(gw.agents[ag].loc[0] - t_loc[0])
            y_dist = abs(gw.agents[ag].loc[1] - t_loc[1])
            total_dist = x_dist + y_dist

            if total_dist < 4:
                counterfactuals[a_id][t_id] = 1

        # Make sure far away agent does not receive a bad counterfactual
        if 1 not in counterfactuals[a_id]:
            counterfactuals[a_id] = [1 for i in range(len(gw.targets))]

    return counterfactuals

def calc_cfl_difference(g_reward, gw, counterfactuals):
    """
    Calculate the difference reward for each agent using CFL counterfactuals
    """
    difference_reward = np.zeros(len(gw.agents))

    # Count number of agents at a target
    for i in range(len(gw.agents)):
        counterfactual_global_reward = 0
        target_capture_counter = np.zeros(len(gw.targets))
        for t_id, t_loc in enumerate(gw.targets):
            for a_id, ag in enumerate(gw.agents):
                if t_loc == gw.agents[ag].loc and a_id != i and counterfactuals[a_id][t_id] == 1:
                    target_capture_counter[t_id] += 1

        # If target has at least one agent, targets increases reward
        for tcount in target_capture_counter:
            if tcount > 0:
                counterfactual_global_reward += gw.reward

        difference_reward[i] = g_reward - counterfactual_global_reward

    return difference_reward
