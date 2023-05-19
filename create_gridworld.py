from gridworld import GridWorld


def create_gridworld(n_agents, n_targets, width, height, n_steps):
    """
    Create Gridworld configuration files
    """
    gw = GridWorld(width, height)
    # gw.create_world(n_agents, n_targets)  # Create a new world configuration
    gw.create_center_world(n_agents, n_targets, n_steps)


if __name__ == "__main__":
    width = 8
    height = 8
    n_targets = 15
    n_agents = 15
    n_steps = 10

    create_gridworld(n_agents, n_targets, width, height, n_steps)
