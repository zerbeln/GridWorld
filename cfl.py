import numpy as np


def create_counterfactuals(gw, nsteps):
    """
    Create counterfactuals that agents will use for CFL learning
    """
    counterfactuals = [[0 for i in range(len(gw.targets))] for j in range(len(gw.agents))]
    target_distances = [[0 for i in range(len(gw.targets))] for j in range(len(gw.agents))]
    for a_id, ag in enumerate(gw.agents):
        for t_id, t_loc in enumerate(gw.targets):
            x_dist = abs(gw.agents[ag].loc[0] - t_loc[0])
            y_dist = abs(gw.agents[ag].loc[1] - t_loc[1])
            total_dist = x_dist + y_dist

            if total_dist < (0.5*nsteps):
                counterfactuals[a_id][t_id] = 1

        # print(counterfactuals[a_id])
        # Make sure far away agent does not receive a bad counterfactual
        if 1 not in counterfactuals[a_id]:
            for t_id in range(len(gw.targets)):
                counterfactuals[a_id][t_id] = 1

    return counterfactuals


def calc_cfl_difference(g_reward, gw, counterfactuals):
    """
    Calculate the difference reward for each agent using CFL counterfactuals
    """
    difference_reward = np.zeros(len(gw.agents))

    # Count number of agents at a target
    for i in range(len(gw.agents)):
        target_capture_counter = np.zeros(len(gw.targets))
        for t_id, t_loc in enumerate(gw.targets):
            for a_id, ag in enumerate(gw.agents):
                if t_loc == gw.agents[ag].loc and a_id != i and counterfactuals[i][t_id] == 1:  # Null counterfactual
                    target_capture_counter[t_id] += 1
                elif t_loc == gw.agents[ag].loc and counterfactuals[i][t_id] == 0:  # Counterfactual matches action
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
