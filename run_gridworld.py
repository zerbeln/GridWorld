from gridworld import GridWorld
from difference_reward import calc_difference_reward
from pbrs import PBRS


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


def q_learning_gridworld(gw, n_agents, n_targets, n_epochs, n_steps):
    """
    Use a standard q-learning approach to solve a multiagent gridworld
    """
    for ep in range(n_epochs):
        for ag in gw.agents:
            gw.agents[ag].reset_agent()
            agent_state = gw.agents[ag].loc[0] + gw.height * gw.agents[ag].loc[1]
            gw.agents[ag].set_current_state(agent_state)

        for t in range(n_steps):
            for ag in gw.agents:
                agent_state = gw.agents[ag].loc[0] + gw.height * gw.agents[ag].loc[1]
                gw.agents[ag].action = gw.agents[ag].get_egreedy_action(agent_state)
                reward, gw.agents[ag].loc = gw.step(gw.agents[ag].loc, gw.agents[ag].action)
                next_state = gw.agents[ag].loc[0] + gw.height * gw.agents[ag].loc[1]
                gw.agents[ag].update_state(next_state)

                # Update Q-Table
                gw.agents[ag].update_q_val(reward)

                if reward > 0:
                    break

    # Record agent solutions
    solution = [[] for ag in range(n_agents)]
    for ag in gw.agents:
        gw.agents[ag].reset_agent()
        agent_state = gw.agents[ag].loc[0] + gw.height * gw.agents[ag].loc[1]
        gw.agents[ag].set_current_state(agent_state)
    for t in range(n_steps):
        for ag in range(n_agents):
            agent_state = gw.agents[f'A{ag}'].loc[0] + gw.height * gw.agents[f'A{ag}'].loc[1]
            gw.agents[f'A{ag}'].action = gw.agents[f'A{ag}'].get_greedy_action(agent_state)
            solution[ag].append(gw.agents[f'A{ag}'].action)
            reward, gw.agents[f'A{ag}'].loc = gw.step(gw.agents[f'A{ag}'].loc, gw.agents[f'A{ag}'].action)
            next_state = gw.agents[f'A{ag}'].loc[0] + gw.height * gw.agents[f'A{ag}'].loc[1]
            gw.agents[f'A{ag}'].update_state(next_state)

    print("Agent Start Position: ", gw.agents['A0'].initial_position)
    print("Target Location: ", gw.targets[0])
    print("Solution: ", solution)


def gridworld_global(gw, n_agents, n_targets, n_epochs, n_steps):
    """
    Train multiagent team on Gridworld using global reward as feedback
    """
    for ep in range(n_epochs):
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

    # Record agent solutions
    solution = [[] for ag in range(n_agents)]
    for ag in gw.agents:
        gw.agents[ag].reset_agent()
        agent_state = gw.agents[ag].loc[0] + gw.height * gw.agents[ag].loc[1]
        gw.agents[ag].set_current_state(agent_state)
    for t in range(n_steps):
        for ag in range(n_agents):
            agent_state = gw.agents[f'A{ag}'].loc[0] + gw.height * gw.agents[f'A{ag}'].loc[1]
            gw.agents[f'A{ag}'].action = gw.agents[f'A{ag}'].get_greedy_action(agent_state)
            solution[ag].append(gw.agents[f'A{ag}'].action)
            l_reward, gw.agents[f'A{ag}'].loc = gw.step(gw.agents[f'A{ag}'].loc, gw.agents[f'A{ag}'].action)
            next_state = gw.agents[f'A{ag}'].loc[0] + gw.height * gw.agents[f'A{ag}'].loc[1]
            gw.agents[f'A{ag}'].update_state(next_state)


def gridworld_difference(gw, n_agents, n_targets, n_epochs, n_steps):
    """
    Train multiagent team on Gridworld using difference reward as feedback
    """
    for ep in range(n_epochs):
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
            for ag in range(n_agents):
                gw.agents[f'A{ag}'].update_q_val(d_reward[ag])

    # Record agent solutions
    solution = [[] for ag in range(n_agents)]
    for ag in gw.agents:
        gw.agents[ag].reset_agent()
        agent_state = gw.agents[ag].loc[0] + gw.height * gw.agents[ag].loc[1]
        gw.agents[ag].set_current_state(agent_state)
    for t in range(n_steps):
        for ag in range(n_agents):
            agent_state = gw.agents[f'A{ag}'].loc[0] + gw.height * gw.agents[f'A{ag}'].loc[1]
            gw.agents[f'A{ag}'].action = gw.agents[f'A{ag}'].get_greedy_action(agent_state)
            solution[ag].append(gw.agents[f'A{ag}'].action)
            l_reward, gw.agents[f'A{ag}'].loc = gw.step(gw.agents[f'A{ag}'].loc, gw.agents[f'A{ag}'].action)
            next_state = gw.agents[f'A{ag}'].loc[0] + gw.height * gw.agents[f'A{ag}'].loc[1]
            gw.agents[f'A{ag}'].update_state(next_state)


def gridworld_pbrs(gw, n_agents, n_targets, n_epochs, n_steps):
    """
    Train multiagent team on Gridworld using potential-based reward shaping
    """
    agent_pbrs = {f'P{ag}': PBRS(gw.n_states) for ag in range(n_agents)}
    for ag in agent_pbrs:
        agent_pbrs[ag].set_potentials(gw)
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
            for ag in range(n_agents):
                # Calculate change in potential
                ag_current_state = gw.agents[f'A{ag}'].current_state  # Current agent state
                ag_prev_state = gw.agents[f'A{ag}'].prev_state  # Previous agent state
                delta_phi = agent_pbrs[f'P{ag}'].potential_function(ag_current_state, ag_prev_state)

                # Update Q-table and state potentials
                gw.agents[f'A{ag}'].update_q_val(g_reward + delta_phi)

    # Record agent solutions
    solution = [[] for ag in range(n_agents)]
    for ag in gw.agents:
        gw.agents[ag].reset_agent()
        agent_state = gw.agents[ag].loc[0] + gw.height * gw.agents[ag].loc[1]
        gw.agents[ag].set_current_state(agent_state)
    for t in range(n_steps):
        for ag in range(n_agents):
            agent_state = gw.agents[f'A{ag}'].loc[0] + gw.height * gw.agents[f'A{ag}'].loc[1]
            gw.agents[f'A{ag}'].action = gw.agents[f'A{ag}'].get_greedy_action(agent_state)
            solution[ag].append(gw.agents[f'A{ag}'].action)
            l_reward, gw.agents[f'A{ag}'].loc = gw.step(gw.agents[f'A{ag}'].loc, gw.agents[f'A{ag}'].action)
            next_state = gw.agents[f'A{ag}'].loc[0] + gw.height * gw.agents[f'A{ag}'].loc[1]
            gw.agents[f'A{ag}'].update_state(next_state)


if __name__ == "__main__":
    width = 5
    height = 5
    n_agents = 1
    n_targets = 1
    n_epochs = 10
    n_steps = 15

    gw = GridWorld(width, height)
    gw.create_world(n_agents, n_targets)

    # Training with Global Reward
    # gridworld_global(gw, n_agents, n_targets, n_epochs, n_steps)

    # Training with Difference Reward
    # gridworld_difference(gw, n_agents, n_targets, n_epochs, n_steps)
    gridworld_pbrs(gw, n_agents, n_targets, n_epochs, n_steps)

    # TODO: Implement CFL
