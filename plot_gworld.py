from gridworld import GridWorld
import matplotlib.pyplot as plt


def create_plot_gridworld(n_agents, n_targets, width, height):
    gw = GridWorld(width, height)
    gw.load_configuration(n_agents, n_targets)  # Load GridWorld configuration from CSV files

    agent_x = []
    agent_y = []
    for ag in gw.agents:
        agent_x.append(gw.agents[ag].loc[0])
        agent_y.append(gw.agents[ag].loc[1])

    target_x = []
    target_y = []
    for t_loc in gw.targets:
        target_x.append(t_loc[0])
        target_y.append(t_loc[1])

    plt.scatter(target_x, target_y)
    plt.scatter(agent_x, agent_y)
    plt.xlim([0, width-1])
    plt.ylim([0, height-1])
    plt.legend(["Targets", "Agents"])
    plt.show()



