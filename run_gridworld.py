from gridworld import GridWorld
from difference_reward import calc_difference_reward
from cfl import calc_cfl_difference, create_counterfactuals
from pbrs import PBRS
import numpy as np
from global_functions import create_pickle_file, create_csv_file


def manual_gridworld():
    """
    This is a manually written gridworld solver to test environmental mechanics (for a single agent gridworld)
    """
    width = 5
    height = 5
    n_agents = 1
    n_targets = 1

    gw = GridWorld(width, height)
    gw.create_world(n_agents, n_targets)

    # Testing environment mechanics with manual strategy
    x_dist = gw.targets[0][0] - gw.agents['A0'].loc[0]
    y_dist = gw.targets[0][1] - gw.agents['A0'].loc[1]

    solution = []
    while y_dist != 0:
        if y_dist > 0:
            action = 0
            solution.append(action)
            reward, gw.agents['A0'].loc = gw.step(gw.agents['A0'].loc, action)
        else:
            action = 1
            solution.append(action)
            reward, gw.agents['A0'].loc = gw.step(gw.agents['A0'].loc, action)
        y_dist = gw.targets[0][1] - gw.agents['A0'].loc[1]

    while x_dist != 0:
        if x_dist < 0:
            action = 2
            solution.append(action)
            reward, gw.agents['A0'].loc = gw.step(gw.agents['A0'].loc, action)
        else:
            action = 3
            solution.append(action)
            reward, gw.agents['A0'].loc = gw.step(gw.agents['A0'].loc, action)
        x_dist = gw.targets[0][0] - gw.agents['A0'].loc[0]

    return solution


def q_learning_gridworld(gw, n_agents, stat_runs, n_epochs, n_steps):
    """
    Use a standard q-learning approach to solve a multiagent gridworld
    """
    q_learning_curve = np.zeros((n_agents, stat_runs, n_epochs))
    g_learning_curve = np.zeros((stat_runs, n_epochs))
    for sr in range(stat_runs):
        # Zero out the Q-Table of the agent for the new stat run
        for ag in gw.agents:
            gw.agents[ag].reset_learner()
        best_solution = [[] for ag in range(n_agents)]
        for ep in range(n_epochs):
            # Reset agent to initial conditions (does not erase Q-Table)
            for ag in gw.agents:
                gw.agents[ag].reset_agent()
                agent_state = gw.agents[ag].loc[0] + gw.height * gw.agents[ag].loc[1]
                gw.agents[ag].set_current_state(agent_state)

            # Take actions over n timesteps
            for t in range(n_steps):
                for ag in gw.agents:
                    agent_state = gw.agents[ag].loc[0] + gw.height * gw.agents[ag].loc[1]
                    gw.agents[ag].action = gw.agents[ag].get_egreedy_action(agent_state)
                    l_reward, gw.agents[ag].loc = gw.step(gw.agents[ag].loc, gw.agents[ag].action)
                    next_state = gw.agents[ag].loc[0] + gw.height * gw.agents[ag].loc[1]
                    gw.agents[ag].update_state(next_state)

                    # Update Q-Table
                    gw.agents[ag].update_q_val(l_reward)

            # Test agent's best solution thus far
            for ag in gw.agents:
                gw.agents[ag].reset_agent()
                agent_state = gw.agents[ag].loc[0] + gw.height * gw.agents[ag].loc[1]
                gw.agents[ag].set_current_state(agent_state)
            for t in range(n_steps):
                for id, ag in enumerate(gw.agents):
                    agent_state = gw.agents[ag].loc[0] + gw.height * gw.agents[ag].loc[1]
                    gw.agents[ag].action = gw.agents[ag].get_greedy_action(agent_state)
                    if ep % (n_epochs-1) == 0:
                        best_solution[id].append(gw.agents[ag].action)
                    l_reward, gw.agents[ag].loc = gw.step(gw.agents[ag].loc, gw.agents[ag].action)
                    q_learning_curve[id, sr, ep] += l_reward
            g_reward = gw.calculate_g_reward()
            g_learning_curve[sr, ep] = g_reward

        create_csv_file(best_solution, "Output_Data/", "QLearningAgentSolutions.csv")
    create_pickle_file(q_learning_curve, "Output_Data/", "QLearningReward")
    create_pickle_file(g_learning_curve, "Output_Data/", "QLearning_GReward")


def gridworld_global(gw, n_agents, stat_runs, n_epochs, n_steps):
    """
    Train multiagent team on Gridworld using global reward as feedback
    """
    agent_learning_curves = np.zeros((stat_runs, n_epochs))
    for sr in range(stat_runs):
        best_solution = [[] for ag in range(n_agents)]
        # Zero out the Q-Table of the agent for the new stat run
        for ag in gw.agents:
            gw.agents[ag].reset_learner()
        for ep in range(n_epochs):
            # Reset agent to initial conditions (does not erase Q-Table)
            for ag in gw.agents:
                gw.agents[ag].reset_agent()
                agent_state = gw.agents[ag].loc[0] + gw.height * gw.agents[ag].loc[1]
                gw.agents[ag].set_current_state(agent_state)

            # Agents choose actions for pre-determined number of time steps
            for t in range(n_steps):
                for ag in gw.agents:
                    agent_state = gw.agents[ag].loc[0] + gw.height * gw.agents[ag].loc[1]
                    gw.agents[ag].action = gw.agents[ag].get_egreedy_action(agent_state)
                    l_reward, gw.agents[ag].loc = gw.step(gw.agents[ag].loc, gw.agents[ag].action)
                    next_state = gw.agents[ag].loc[0] + gw.height * gw.agents[ag].loc[1]
                    gw.agents[ag].update_state(next_state)

                g_reward = gw.calculate_g_reward()
                # Update Agent Q-Tables
                for ag in gw.agents:
                    gw.agents[ag].update_q_val(g_reward)

            # Test agent solution
            for ag in gw.agents:
                gw.agents[ag].reset_agent()
                agent_state = gw.agents[ag].loc[0] + gw.height * gw.agents[ag].loc[1]
                gw.agents[ag].set_current_state(agent_state)
            for t in range(n_steps):
                for id, ag in enumerate(gw.agents):
                    agent_state = gw.agents[ag].loc[0] + gw.height * gw.agents[ag].loc[1]
                    gw.agents[ag].action = gw.agents[ag].get_greedy_action(agent_state)
                    if ep % (n_epochs-1) == 0:
                        best_solution[id].append(gw.agents[ag].action)
                    l_reward, gw.agents[ag].loc = gw.step(gw.agents[ag].loc, gw.agents[ag].action)
            g_reward = gw.calculate_g_reward()
            agent_learning_curves[sr, ep] = g_reward

    create_pickle_file(agent_learning_curves, "Output_Data/", "Global_Rewards")


def gridworld_difference(gw, n_agents, stat_runs, n_epochs, n_steps):
    """
    Train multiagent team on Gridworld using difference reward as feedback
    """
    agent_learning_curves = np.zeros((stat_runs, n_epochs))
    for sr in range(stat_runs):
        best_solution = [[] for ag in range(n_agents)]
        # Zero out the Q-Table of the agent for the new stat run
        for ag in gw.agents:
            gw.agents[ag].reset_learner()
        for ep in range(n_epochs):
            # Reset agent to initial conditions (does not erase Q-Table)
            for ag in gw.agents:
                gw.agents[ag].reset_agent()
                agent_state = gw.agents[ag].loc[0] + gw.height * gw.agents[ag].loc[1]
                gw.agents[ag].set_current_state(agent_state)

            # Agents choose actions for pre-determined number of time steps
            for t in range(n_steps):
                for ag in gw.agents:
                    agent_state = gw.agents[ag].loc[0] + gw.height * gw.agents[ag].loc[1]
                    gw.agents[ag].action = gw.agents[ag].get_egreedy_action(agent_state)
                    l_reward, gw.agents[ag].loc = gw.step(gw.agents[ag].loc, gw.agents[ag].action)
                    next_state = gw.agents[ag].loc[0] + gw.height * gw.agents[ag].loc[1]
                    gw.agents[ag].update_state(next_state)

                g_reward = gw.calculate_g_reward()
                d_reward = calc_difference_reward(g_reward, gw)
                # Update Agent Q-Tables
                for id, ag in enumerate(gw.agents):
                    gw.agents[ag].update_q_val(d_reward[id])

            # Test agent solution
            for ag in gw.agents:
                gw.agents[ag].reset_agent()
                agent_state = gw.agents[ag].loc[0] + gw.height * gw.agents[ag].loc[1]
                gw.agents[ag].set_current_state(agent_state)
            for t in range(n_steps):
                for id, ag in enumerate(gw.agents):
                    agent_state = gw.agents[ag].loc[0] + gw.height * gw.agents[ag].loc[1]
                    gw.agents[ag].action = gw.agents[ag].get_greedy_action(agent_state)
                    if ep % (n_epochs-1) == 0:
                        best_solution[id].append(gw.agents[ag].action)
                    l_reward, gw.agents[ag].loc = gw.step(gw.agents[ag].loc, gw.agents[ag].action)
            g_reward = gw.calculate_g_reward()
            agent_learning_curves[sr, ep] = g_reward

    create_pickle_file(agent_learning_curves, "Output_Data/", "Difference_Rewards")


def gridworld_pbrs(gw, n_agents, stat_runs, n_epochs, n_steps):
    """
    Train multiagent team on Gridworld using potential-based reward shaping
    """
    agent_learning_curves = np.zeros((stat_runs, n_epochs))
    agent_pbrs = {f'P{ag}': PBRS(gw.n_states) for ag in range(n_agents)}
    for id, ag in enumerate(agent_pbrs):
        agent_pbrs[ag].set_potentials(gw, id, n_steps)

    for sr in range(stat_runs):
        best_solution = [[] for ag in range(n_agents)]
        # Zero out the Q-Table of the agent for the new stat run
        for ag in range(n_agents):
            gw.agents[f'A{ag}'].reset_learner()
        for ep in range(n_epochs):
            # Reset agents to initial conditions
            for ag in gw.agents:
                gw.agents[ag].reset_agent()
                agent_state = gw.agents[ag].loc[0] + gw.height * gw.agents[ag].loc[1]
                gw.agents[ag].set_current_state(agent_state)

            # Agents choose actions for pre-determined number of time steps
            for t in range(n_steps):
                for ag in gw.agents:
                    agent_state = gw.agents[ag].loc[0] + gw.height * gw.agents[ag].loc[1]
                    gw.agents[ag].action = gw.agents[ag].get_egreedy_action(agent_state)
                    l_reward, gw.agents[ag].loc = gw.step(gw.agents[ag].loc, gw.agents[ag].action)
                    next_state = gw.agents[ag].loc[0] + gw.height * gw.agents[ag].loc[1]
                    gw.agents[ag].update_state(next_state)

                # Calculate agent rewards
                g_reward = gw.calculate_g_reward()
                for id, ag in enumerate(gw.agents):
                    # Calculate change in potential
                    ag_current_state = gw.agents[ag].current_state  # Current agent state
                    ag_prev_state = gw.agents[ag].prev_state  # Previous agent state
                    delta_phi = agent_pbrs[f'P{id}'].potential_function(ag_current_state, ag_prev_state)

                    # Update Q-table and state potentials
                    gw.agents[ag].update_q_val(g_reward + delta_phi)

            # Test agent solution
            for ag in gw.agents:
                gw.agents[ag].reset_agent()
                agent_state = gw.agents[ag].loc[0] + gw.height * gw.agents[ag].loc[1]
                gw.agents[ag].set_current_state(agent_state)
            for t in range(n_steps):
                for id, ag in enumerate(gw.agents):
                    agent_state = gw.agents[ag].loc[0] + gw.height * gw.agents[ag].loc[1]
                    gw.agents[ag].action = gw.agents[ag].get_greedy_action(agent_state)
                    if ep % (n_epochs-1) == 0:
                        best_solution[id].append(gw.agents[ag].action)
                    l_reward, gw.agents[ag].loc = gw.step(gw.agents[ag].loc, gw.agents[ag].action)
            g_reward = gw.calculate_g_reward()
            agent_learning_curves[sr, ep] = g_reward

    create_pickle_file(agent_learning_curves, "Output_Data/", "PBRS_Rewards")


def gridworld_cfrl(gw, n_agents, stat_runs, n_epochs, n_steps, counterfactuals):
    """
    Train multiagent team on Gridworld using CFL difference rewards as feedback
    """
    agent_learning_curves = np.zeros((stat_runs, n_epochs))
    for sr in range(stat_runs):
        best_solution = [[] for ag in range(n_agents)]
        # Zero out the Q-Table of the agent for the new stat run
        for ag in gw.agents:
            gw.agents[ag].reset_learner()
        for ep in range(n_epochs):
            # Reset agent to initial conditions (does not erase Q-Table)
            for ag in gw.agents:
                gw.agents[ag].reset_agent()
                agent_state = gw.agents[ag].loc[0] + gw.height * gw.agents[ag].loc[1]
                gw.agents[ag].set_current_state(agent_state)

            # Agents choose actions for pre-determined number of time steps
            for t in range(n_steps):
                for ag in gw.agents:
                    agent_state = gw.agents[ag].loc[0] + gw.height * gw.agents[ag].loc[1]
                    gw.agents[ag].action = gw.agents[ag].get_egreedy_action(agent_state)
                    l_reward, gw.agents[ag].loc = gw.step(gw.agents[ag].loc, gw.agents[ag].action)
                    next_state = gw.agents[ag].loc[0] + gw.height * gw.agents[ag].loc[1]
                    gw.agents[ag].update_state(next_state)

                g_reward = gw.calculate_g_reward()
                d_reward = calc_cfl_difference(g_reward, gw, counterfactuals)
                # Update Agent Q-Tables
                for id, ag in enumerate(gw.agents):
                    gw.agents[ag].update_q_val(d_reward[id])

            # Test agent solution
            for ag in gw.agents:
                gw.agents[ag].reset_agent()
                agent_state = gw.agents[ag].loc[0] + gw.height * gw.agents[ag].loc[1]
                gw.agents[ag].set_current_state(agent_state)
            for t in range(n_steps):
                for id, ag in enumerate(gw.agents):
                    if gw.agents[ag].loc not in gw.targets:
                        agent_state = gw.agents[ag].loc[0] + gw.height * gw.agents[ag].loc[1]
                        gw.agents[ag].action = gw.agents[ag].get_greedy_action(agent_state)
                        if ep % (n_epochs - 1) == 0:
                            best_solution[id].append(gw.agents[ag].action)
                        l_reward, gw.agents[ag].loc = gw.step(gw.agents[ag].loc, gw.agents[ag].action)
            g_reward = gw.calculate_g_reward()
            agent_learning_curves[sr, ep] = g_reward

    create_pickle_file(agent_learning_curves, "Output_Data/", "CFL_Rewards")


if __name__ == "__main__":
    width = 8
    height = 8
    n_agents = 12
    n_targets = 12
    stat_runs = 30
    n_epochs = 200
    n_steps = 10

    gw = GridWorld(width, height)
    # gw.create_world(n_agents, n_targets)  # Create a new world configuration
    gw.load_configuration(n_agents, n_targets)  # Reuse a world configuration

    # Training with Standard Q-Learning and Local Rewaerd
    print("Running Gridworld with Q-Learning Local Reward")
    q_learning_gridworld(gw, n_agents, stat_runs, n_epochs, n_steps)

    # Training with Global Reward
    print("Running Gridworld with Global Reward")
    gridworld_global(gw, n_agents, stat_runs, n_epochs, n_steps)

    # Training with Difference Reward
    print("Running Gridworld with Difference Reward")
    gridworld_difference(gw, n_agents, stat_runs, n_epochs, n_steps)

    # Training with Potential Based Reward Shaping
    print("Running Gridworld with PBRS")
    gridworld_pbrs(gw, n_agents, stat_runs, n_epochs, n_steps)

    print("Running Gridworld with CFL")
    counterfactuals = create_counterfactuals(gw)
    gridworld_cfrl(gw, n_agents, stat_runs, n_epochs, n_steps, counterfactuals)
