from gridworld import GridWorld


def manual_gridworld(gw):
    """
    This is a manually written gridworld solver to test environmental mechanics
    """

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


if __name__ == "__main__":
    width = 5
    height = 5
    n_agents = 1
    n_targets = 1

    gw = GridWorld(width, height)
    gw.create_world(n_agents, n_targets)

    n_epochs = 500
    n_steps = 15
    for ep in range(n_epochs):
        gw.agents['A0'].reset_agent()

        for t in range(n_steps):
            agent_state = gw.agents['A0'].loc[0] + gw.height*gw.agents['A0'].loc[1]
            action = gw.agents['A0'].get_egreedy_action(agent_state)
            reward, gw.agents['A0'].loc = gw.step(gw.agents['A0'].loc, action)
            next_state = gw.agents['A0'].loc[0] + gw.height*gw.agents['A0'].loc[1]
            gw.agents['A0'].update_q_val(agent_state, action, next_state, reward)

            if reward > 0:
                break

    gw.agents['A0'].reset_agent()
    solution = []
    for t in range(n_steps):
        agent_state = gw.agents['A0'].loc[0] + gw.height * gw.agents['A0'].loc[1]
        action = gw.agents['A0'].get_greedy_action(agent_state)
        solution.append(action)
        reward, gw.agents['A0'].loc = gw.step(gw.agents['A0'].loc, action)
        next_state = gw.agents['A0'].loc[0] + gw.height * gw.agents['A0'].loc[1]
        gw.agents['A0'].update_q_val(agent_state, action, next_state, reward)

        if reward > 0:
            break

    print("Agent Start Position: ", gw.agents['A0'].initial_position)
    print("Target Location: ", gw.targets[0])
    print("Solution: ", solution)
