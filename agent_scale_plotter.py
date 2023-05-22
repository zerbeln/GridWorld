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


def create_scaling_plot(n_tests, size, x_axis):
    # Plot Color Palette
    color1 = np.array([26, 133, 255]) / 255  # Blue
    color2 = np.array([255, 194, 10]) / 255  # Yellow
    color3 = np.array([230, 97, 0]) / 255  # Orange
    color4 = np.array([93, 58, 155]) / 255  # Purple
    color5 = np.array([211, 95, 183]) / 255  # Fuschia

    # Q-Learning Data -------------------------------------------------------------
    ql_rewards = []
    ql_data = []

    # Global Reward Data ---------------------------------------------------------
    g_rewards = []
    g_data = []

    # Difference Reward Data --------------------------------------------------------
    d_rewards = []
    d_data = []

    # PBRS Data -----------------------------------------------------------------------
    pbrs_rewards = []
    pbrs_data = []

    # CFL Data ------------------------------------------------------------------------
    cfl_rewards = []
    cfl_data = []

    # DRiP Data ------------------------------------------------------------------------
    drip_rewards = []
    drip_data = []

    # CFL-P Data ------------------------------------------------------------------------
    cflp_rewards = []
    cflp_data = []

    # Sort the data
    for na in x_axis:
        ql_data.append(import_pickle_data(f"{na}Agents/Output_Data/QLearning_GReward"))
        g_data.append(import_pickle_data(f"{na}Agents/Output_Data/Global_Rewards"))
        d_data.append(import_pickle_data(f"{na}Agents/Output_Data/Difference_Rewards"))
        pbrs_data.append(import_pickle_data(f"{na}Agents/Output_Data/PBRS_Rewards"))
        cfl_data.append(import_pickle_data(f"{na}Agents/Output_Data/CFL_Rewards"))
        drip_data.append(import_pickle_data(f"{na}Agents/Output_Data/DRIP_Rewards"))
        cflp_data.append(import_pickle_data(f"{na}Agents/Output_Data/CFLP_Rewards"))

    # Average the data
    for i in range(n_tests):
        ql_avg_data = np.mean(ql_data[i], axis=0)
        ql_rewards.append(ql_avg_data[-1])
        g_avg_data = np.mean(g_data[i][:], axis=0)
        g_rewards.append(g_avg_data[-1])
        d_avg_data = np.mean(d_data[i][:], axis=0)
        d_rewards.append(d_avg_data[-1])
        pbrs_avg_data = np.mean(pbrs_data[i][:], axis=0)
        pbrs_rewards.append(pbrs_avg_data[-1])
        cfl_avg_data = np.mean(cfl_data[i][:], axis=0)
        cfl_rewards.append(cfl_avg_data[-1])
        drip_avg_data = np.mean(drip_data[i][:], axis=0)
        drip_rewards.append(drip_avg_data[-1])
        cflp_avg_data = np.mean(cflp_data[i][:], axis=0)
        cflp_rewards.append(cflp_avg_data[-1])

    # Make Plot -------------------------------------------------------------------------
    plt.plot(x_axis, ql_rewards, '-s', color='black')
    plt.plot(x_axis, g_rewards, '-^', color=color2)
    plt.plot(x_axis, d_rewards, '-P', color=color3)
    plt.plot(x_axis, pbrs_rewards, '-o', color=color5)
    plt.plot(x_axis, cfl_rewards, '-*', color=color4)
    plt.plot(x_axis, drip_rewards, color="limegreen")
    plt.plot(x_axis, cflp_rewards, color=color1)

    # Graph Details
    plt.xlabel("Number of Agents/Targets")
    plt.ylabel("Average Percentage of Targets Captured")
    plt.ylim([0, 100])
    plt.legend(["Q-Learning", "Global", "Difference", "PBRS", "CFL", "DRiP", "CFL-P"], ncol=2, bbox_to_anchor=(0.01, 0.99), loc='upper left', borderaxespad=0)

    # Save the plot
    if not os.path.exists('Plots'):  # If Data directory does not exist, create it
        os.makedirs('Plots')
    plt.savefig(f'Plots/{size}x{size}_AgentScaling.pdf')

    # Show the plot
    plt.show()


if __name__ == "__main__":
    n_epochs = 3000
    size = int(sys.argv[1])
    x_axis = []
    if size == 8:
        x_axis = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    elif size == 10:
        x_axis = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    n_tests = len(x_axis)

    create_scaling_plot(n_tests, size, x_axis)
