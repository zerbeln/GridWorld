from gridworld import GridWorld
from plot_gworld import create_plot_gridworld


def create_gridworld(n_agents, n_targets, width, height, n_steps):
    """
    Create Gridworld configuration files
    """
    gw = GridWorld(width, height)
    # gw.create_world(n_agents, n_targets)  # Create a new world configuration
    gw.create_center_world(n_agents, n_targets, n_steps)


if __name__ == "__main__":
    width = 10
    height = 10
    n_agents = 20
    n_targets = n_agents
    n_steps = 10

    create_gridworld(n_agents, n_targets, width, height, n_steps)
    create_plot_gridworld(n_agents, n_targets, width, height)
