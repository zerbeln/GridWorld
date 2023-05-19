import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import sys
import matplotlib.colors as mcolors


def import_pickle_data(file_path):
    """
    Load saved Neural Network policies from pickle file
    """

    data_file = open(file_path, 'rb')
    pickle_data = pickle.load(data_file)
    data_file.close()

    return pickle_data


def create_q_learn_plot(n_agents, n_epochs):
    """
    Plot the individual performance of agents learning with local rewards and standard Q-Learning
    """
    # Q-Learning Data
    q_learn_data = import_pickle_data("Output_Data/QLearningReward")
    q_learn_reward = []
    for ag in range(n_agents):
        q_learn_reward.append(np.mean(q_learn_data[ag, :], axis=0))

    x_axis = [i for i in range(n_epochs)]

    for ag in range(n_agents):
        plt.plot(x_axis, q_learn_reward[0])

    # Graph Details
    plt.xlabel("Epochs")
    plt.ylabel("Agent Reward")
    plt.legend(["Q-Learning"])

    plt.show()


def create_learning_curve(n_agents, n_epochs, size):
    # Plot Color Palette
    color1 = np.array([26, 133, 255]) / 255  # Blue
    color2 = np.array([255, 194, 10]) / 255  # Yellow
    color3 = np.array([230, 97, 0]) / 255  # Orange
    color4 = np.array([93, 58, 155]) / 255  # Purple
    color5 = np.array([211, 95, 183]) / 255  # Fuschia

    # Q-Learning Data
    ql_rdata = import_pickle_data("Output_Data/QLearning_GReward")
    ql_rewards = np.mean(ql_rdata[:], axis=0)

    # Global Reward Data
    g_rdata = import_pickle_data("Output_Data/Global_Rewards")
    g_rewards = np.mean(g_rdata[:], axis=0)

    # Difference Reward Data
    d_rdata = import_pickle_data("Output_Data/Difference_Rewards")
    d_rewards = np.mean(d_rdata[:], axis=0)

    # PBRS Data
    pbrs_rdata = import_pickle_data("Output_Data/PBRS_Rewards")
    pbrs_rewards = np.mean(pbrs_rdata[:], axis=0)

    # CFL Data
    cfl_rdata = import_pickle_data("Output_Data/CFL_Rewards")
    cfl_rewards = np.mean(cfl_rdata[:], axis=0)

    # DRIP Data
    drip_rdata = import_pickle_data("Output_Data/DRIP_Rewards")
    drip_rewards = np.mean(drip_rdata[:], axis=0)

    # CFLP Data
    cflp_rdata = import_pickle_data("Output_Data/CFLP_Rewards")
    cflp_rewards = np.mean(cflp_rdata[:], axis=0)

    x_axis = [i for i in range(n_epochs)]
    plt.plot(x_axis, ql_rewards, color='black')
    plt.plot(x_axis, g_rewards, color=color2)
    plt.plot(x_axis, d_rewards, color=color3)
    plt.plot(x_axis, pbrs_rewards, color=color5)
    plt.plot(x_axis, cfl_rewards, color=color4)
    plt.plot(x_axis, drip_rewards, color="limegreen")
    plt.plot(x_axis, cflp_rewards, color=color1)

    # Graph Details
    plt.xlabel("Number of Epochs")
    plt.ylabel("Average Team Reward")
    plt.ylim([0, 100])
    plt.legend(["Q-Learning", "Global", "Difference", "PBRS", "CFL", "DRiP", "CFL-P"], ncol=2, bbox_to_anchor=(0.99, 0.01), loc='lower right', borderaxespad=0)

    # Save the plot
    if not os.path.exists('Plots'):  # If Data directory does not exist, create it
        os.makedirs('Plots')
    plt.savefig(f'Plots/{size}x{size}_{n_agents}Agent_LCurves.pdf')

    # Show the plot
    plt.show()


if __name__ == "__main__":
    n_epochs = 2000
    # n_agents = int(sys.argv[1])
    # size = int(sys.argv[2])
    n_agents = 15
    size = 8
    # create_q_learn_plot(n_agents, n_epochs)
    create_learning_curve(n_agents, n_epochs, size)
