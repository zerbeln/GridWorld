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
    if size == 8:
        ql_data.append(import_pickle_data("5Agents/Output_Data/QLearning_GReward"))
        ql_data.append(import_pickle_data("8Agents/Output_Data/QLearning_GReward"))
        ql_data.append(import_pickle_data("10Agents/Output_Data/QLearning_GReward"))
        ql_data.append(import_pickle_data("12Agents/Output_Data/QLearning_GReward"))
        ql_data.append(import_pickle_data("15Agents/Output_Data/QLearning_GReward"))
        ql_data.append(import_pickle_data("18Agents/Output_Data/QLearning_GReward"))
        ql_data.append(import_pickle_data("20Agents/Output_Data/QLearning_GReward"))
    elif size == 10:
        ql_data.append(import_pickle_data("10Agents/Output_Data/QLearning_GReward"))
        ql_data.append(import_pickle_data("15Agents/Output_Data/QLearning_GReward"))
        ql_data.append(import_pickle_data("20Agents/Output_Data/QLearning_GReward"))
        ql_data.append(import_pickle_data("25Agents/Output_Data/QLearning_GReward"))
        ql_data.append(import_pickle_data("30Agents/Output_Data/QLearning_GReward"))
    for i in range(n_tests):
        avg_data = np.mean(ql_data[i], axis=0)
        ql_rewards.append(avg_data[-1])

    # Global Reward Data ---------------------------------------------------------
    g_rewards = []
    g_data = []
    if size == 8:
        g_data.append(import_pickle_data("5Agents/Output_Data/Global_Rewards"))
        g_data.append(import_pickle_data("8Agents/Output_Data/Global_Rewards"))
        g_data.append(import_pickle_data("10Agents/Output_Data/Global_Rewards"))
        g_data.append(import_pickle_data("12Agents/Output_Data/Global_Rewards"))
        g_data.append(import_pickle_data("15Agents/Output_Data/Global_Rewards"))
        g_data.append(import_pickle_data("18Agents/Output_Data/Global_Rewards"))
        g_data.append(import_pickle_data("20Agents/Output_Data/Global_Rewards"))
    elif size == 10:
        g_data.append(import_pickle_data("10Agents/Output_Data/Global_Rewards"))
        g_data.append(import_pickle_data("15Agents/Output_Data/Global_Rewards"))
        g_data.append(import_pickle_data("20Agents/Output_Data/Global_Rewards"))
        g_data.append(import_pickle_data("25Agents/Output_Data/Global_Rewards"))
        g_data.append(import_pickle_data("30Agents/Output_Data/Global_Rewards"))
    for i in range(n_tests):
        avg_data = np.mean(g_data[i][:], axis=0)
        g_rewards.append(avg_data[-1])

    # Difference Reward Data --------------------------------------------------------
    d_rewards = []
    d_data = []
    if size == 8:
        d_data.append(import_pickle_data("5Agents/Output_Data/Difference_Rewards"))
        d_data.append(import_pickle_data("8Agents/Output_Data/Difference_Rewards"))
        d_data.append(import_pickle_data("10Agents/Output_Data/Difference_Rewards"))
        d_data.append(import_pickle_data("12Agents/Output_Data/Difference_Rewards"))
        d_data.append(import_pickle_data("15Agents/Output_Data/Difference_Rewards"))
        d_data.append(import_pickle_data("18Agents/Output_Data/Difference_Rewards"))
        d_data.append(import_pickle_data("20Agents/Output_Data/Difference_Rewards"))
    elif size == 10:
        d_data.append(import_pickle_data("10Agents/Output_Data/Difference_Rewards"))
        d_data.append(import_pickle_data("15Agents/Output_Data/Difference_Rewards"))
        d_data.append(import_pickle_data("20Agents/Output_Data/Difference_Rewards"))
        d_data.append(import_pickle_data("25Agents/Output_Data/Difference_Rewards"))
        d_data.append(import_pickle_data("30Agents/Output_Data/Difference_Rewards"))
    for i in range(n_tests):
        avg_data = np.mean(d_data[i][:], axis=0)
        d_rewards.append(avg_data[-1])

    # PBRS Data -----------------------------------------------------------------------
    pbrs_rewards = []
    pbrs_data = []
    if size == 8:
        pbrs_data.append(import_pickle_data("5Agents/Output_Data/PBRS_Rewards"))
        pbrs_data.append(import_pickle_data("8Agents/Output_Data/PBRS_Rewards"))
        pbrs_data.append(import_pickle_data("10Agents/Output_Data/PBRS_Rewards"))
        pbrs_data.append(import_pickle_data("12Agents/Output_Data/PBRS_Rewards"))
        pbrs_data.append(import_pickle_data("15Agents/Output_Data/PBRS_Rewards"))
        pbrs_data.append(import_pickle_data("18Agents/Output_Data/PBRS_Rewards"))
        pbrs_data.append(import_pickle_data("20Agents/Output_Data/PBRS_Rewards"))
    elif size == 10:
        pbrs_data.append(import_pickle_data("10Agents/Output_Data/PBRS_Rewards"))
        pbrs_data.append(import_pickle_data("15Agents/Output_Data/PBRS_Rewards"))
        pbrs_data.append(import_pickle_data("20Agents/Output_Data/PBRS_Rewards"))
        pbrs_data.append(import_pickle_data("25Agents/Output_Data/PBRS_Rewards"))
        pbrs_data.append(import_pickle_data("30Agents/Output_Data/PBRS_Rewards"))
    for i in range(n_tests):
        avg_data = np.mean(pbrs_data[i][:], axis=0)
        pbrs_rewards.append(avg_data[-1])

    # CFL Data ------------------------------------------------------------------------
    cfl_rewards = []
    cfl_data = []
    if size == 8:
        cfl_data.append(import_pickle_data("5Agents/Output_Data/CFL_Rewards"))
        cfl_data.append(import_pickle_data("8Agents/Output_Data/CFL_Rewards"))
        cfl_data.append(import_pickle_data("10Agents/Output_Data/CFL_Rewards"))
        cfl_data.append(import_pickle_data("12Agents/Output_Data/CFL_Rewards"))
        cfl_data.append(import_pickle_data("15Agents/Output_Data/CFL_Rewards"))
        cfl_data.append(import_pickle_data("18Agents/Output_Data/CFL_Rewards"))
        cfl_data.append(import_pickle_data("20Agents/Output_Data/CFL_Rewards"))
    elif size == 10:
        cfl_data.append(import_pickle_data("10Agents/Output_Data/CFL_Rewards"))
        cfl_data.append(import_pickle_data("15Agents/Output_Data/CFL_Rewards"))
        cfl_data.append(import_pickle_data("20Agents/Output_Data/CFL_Rewards"))
        cfl_data.append(import_pickle_data("25Agents/Output_Data/CFL_Rewards"))
        cfl_data.append(import_pickle_data("30Agents/Output_Data/CFL_Rewards"))
    for i in range(n_tests):
        avg_data = np.mean(cfl_data[i][:], axis=0)
        cfl_rewards.append(avg_data[-1])

    # DRiP Data ------------------------------------------------------------------------
    drip_rewards = []
    drip_data = []
    if size == 8:
        drip_data.append(import_pickle_data("5Agents/Output_Data/DRIP_Rewards"))
        drip_data.append(import_pickle_data("8Agents/Output_Data/DRIP_Rewards"))
        drip_data.append(import_pickle_data("10Agents/Output_Data/DRIP_Rewards"))
        drip_data.append(import_pickle_data("12Agents/Output_Data/DRIP_Rewards"))
        drip_data.append(import_pickle_data("15Agents/Output_Data/DRIP_Rewards"))
        drip_data.append(import_pickle_data("18Agents/Output_Data/DRIP_Rewards"))
        drip_data.append(import_pickle_data("20Agents/Output_Data/CFL_Rewards"))
    elif size == 10:
        drip_data.append(import_pickle_data("10Agents/Output_Data/DRIP_Rewards"))
        drip_data.append(import_pickle_data("15Agents/Output_Data/DRIP_Rewards"))
        drip_data.append(import_pickle_data("20Agents/Output_Data/DRIP_Rewards"))
        drip_data.append(import_pickle_data("25Agents/Output_Data/DRIP_Rewards"))
        drip_data.append(import_pickle_data("30Agents/Output_Data/DRIP_Rewards"))
    for i in range(n_tests):
        avg_data = np.mean(drip_data[i][:], axis=0)
        drip_rewards.append(avg_data[-1])

    # CFL-P Data ------------------------------------------------------------------------
    cflp_rewards = []
    cflp_data = []
    if size == 8:
        cflp_data.append(import_pickle_data("5Agents/Output_Data/CFLP_Rewards"))
        cflp_data.append(import_pickle_data("8Agents/Output_Data/CFLP_Rewards"))
        cflp_data.append(import_pickle_data("10Agents/Output_Data/CFLP_Rewards"))
        cflp_data.append(import_pickle_data("12Agents/Output_Data/CFLP_Rewards"))
        cflp_data.append(import_pickle_data("15Agents/Output_Data/CFLP_Rewards"))
        cflp_data.append(import_pickle_data("18Agents/Output_Data/CFLP_Rewards"))
        cflp_data.append(import_pickle_data("20Agents/Output_Data/CFLp_Rewards"))
    elif size == 10:
        cflp_data.append(import_pickle_data("10Agents/Output_Data/CFLP_Rewards"))
        cflp_data.append(import_pickle_data("15Agents/Output_Data/CFLP_Rewards"))
        cflp_data.append(import_pickle_data("20Agents/Output_Data/CFLP_Rewards"))
        cflp_data.append(import_pickle_data("25Agents/Output_Data/CFLP_Rewards"))
        cflp_data.append(import_pickle_data("30Agents/Output_Data/CFLP_Rewards"))
    for i in range(n_tests):
        avg_data = np.mean(cflp_data[i][:], axis=0)
        cflp_rewards.append(avg_data[-1])

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
    n_epochs = 2000
    size = int(sys.argv[1])
    x_axis = []
    if size == 8:
        n_tests = 7
        x_axis = [4, 5, 6, 7, 8, 9, 10]
    elif size == 10:
        x_axis = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    n_tests = len(x_axis)

    create_scaling_plot(n_tests, size, x_axis)
