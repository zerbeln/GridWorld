from gridworld import GridWorld


def create_gridworld(n_agents, n_targets, width, height):
    """
    Create Gridworld configuration files
    """
    gw = GridWorld(width, height)
    gw.create_world(n_agents, n_targets)  # Create a new world configuration


if __name__ == "__main__":
    width = 8
    height = 8
    n_targets = 12
    n_agents = 12

    create_gridworld(n_agents, n_targets, width, height)
