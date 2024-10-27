import matplotlib.pyplot as plt
import numpy as np
# import pygame
from matplotlib_inline.backend_inline import FigureCanvas
import csv
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from PIL import Image

import copy
import torch


#######################################LRT
# @profile
def calculate_ind_lrt(ind:torch.Tensor, gene_size, number_of_people=60, error=0.001)->float:
    lrts = torch.zeros(gene_size, dtype=torch.double)
    ind = ind.double()

    # Zero MAFs
    nsnp_msk = (ind[:, 1] == 0)
    ind[nsnp_msk, 1] = torch.as_tensor(0.001)

    queried_spns_msk = ind[:, 2] == -1 #Responses
    filtered_ind = ind[~queried_spns_msk]

    genome = filtered_ind[:, 0]
    maf = filtered_ind[:, 1]
    response = filtered_ind[:, 2]

    DN_i = (1 - maf).pow(2 * (number_of_people)) 
    DN_i_1 = (1 - maf).pow(2 * (number_of_people) - 2)

    log1 = torch.log(DN_i) - torch.log(0.001 * DN_i_1)
    log2 = torch.log((0.001 * DN_i_1 * (1 - DN_i))) - torch.log(DN_i * (1 - 0.001 * DN_i_1))

    lrt = (log1+ log2 * response)* genome# + (log3 + log4 * x_hat_i) * (1 - genome)
    lrts[~queried_spns_msk] = lrt
    # print("lrts, lrts.size()", lrts, lrts.size())
    return lrts




def lrt(number_of_people, genome, maf, response):
    genome = torch.as_tensor(genome, dtype=torch.double)
    maf =   torch.as_tensor(maf, dtype=torch.double)
    response =  torch.as_tensor(response, dtype=torch.double)

    if maf == 0: maf = torch.as_tensor(0.001)


    DN_i = (1 - maf).pow(2 * number_of_people) 
    DN_i_1 = (1 - maf).pow(2 * number_of_people - 2)

    # Genome == 1
    log1 = torch.log(DN_i) - torch.log(0.001 * DN_i_1)
    log2 = torch.log((0.001 * DN_i_1 * (1 - DN_i))) - torch.log(DN_i * (1 - 0.001 * DN_i_1))

    lrt = (log1 + log2 * response)* genome #+ (log3  + log4 * x_hat_i) * (1 - genome)

    # print("=================================")
    
    # print(DN_i, DN_i_1)
    # print(log1,log2)
    # print(genome, maf, response, lrt)
    # print("=================================")

    return lrt


def calculate_pvalues(beacon_lrts, control_lrts, b_control_size):
    b_lrts = copy.deepcopy(beacon_lrts)
    c_lrts = copy.deepcopy(control_lrts)

    pvalues=[]
    for blrt in b_lrts:
        pvalue=torch.sum(blrt >= c_lrts) / b_control_size
        pvalues.append(pvalue)
    
    if torch.any(torch.Tensor(pvalues) < 0):
        print(b_lrts, c_lrts)
        assert "Wrong p value"

    return torch.Tensor(pvalues)

#########################################LOG

# def create_csv(results_dir, name):
#     log_env_name = results_dir  + '/logs/' + name + '.csv'
#     print(log_env_name+ "Sssssssssssssss")
#     log_env = csv.writer(open(log_env_name,"w+"))
#     log_env.writerow(["Episode", "Query", "Beacon", "Gene", "SNP", "MAF", "RES", "LRT"])
#     return log_env

# # log tensor data into a CSV file
# def log_env(info, episode, step, log_env):
#     for beacon_idx, beacon_data in enumerate(info):
#         for gene_idx, gene_data in enumerate(beacon_data):
#             snp, maf, res, current, lrts = gene_data
#             # print(res.detach().cpu().numpy(), lrts.detach().cpu().numpy())
#             log_env.writerow([episode, step, beacon_idx, gene_idx, snp.detach().cpu().numpy(), maf.detach().cpu().numpy(), res.detach().cpu().numpy(), lrts.detach().cpu().numpy()])
#     log_env.writerow("--------------------------------------------------------------")



def create_csv(results_dir, name):
    log_env_name = results_dir + '/logs/' + name + '.csv'
    print(log_env_name)
    with open(log_env_name, "w+", newline='') as csvfile:
        log_env_writer = csv.writer(csvfile)
        log_env_writer.writerow(["Episode", "Query", "Beacon", "Gene", "SNP", "MAF", "RES", "LRT"])
    return log_env_name

# log tensor data into a CSV file
def log_env(info, episode, step, log_env_name):
    with open(log_env_name, "a", newline='') as csvfile:
        log_env_writer = csv.writer(csvfile)
        for beacon_idx, beacon_data in enumerate(info):
            for gene_idx, gene_data in enumerate(beacon_data):
                snp, maf, res, current, lrts = gene_data
                log_env_writer.writerow([episode, step, beacon_idx, gene_idx, snp.detach().cpu().numpy(), maf.detach().cpu().numpy(), res.detach().cpu().numpy(), lrts.detach().cpu().numpy()])
        log_env_writer.writerow(["--------------------------------------------------------------"])


# log tensor data into a CSV file
def log_victim(info, episode, step, log_env_name):
    with open(log_env_name, "a", newline='') as csvfile:
        log_env_writer = csv.writer(csvfile)
        for gene_idx, gene_data in enumerate(info):
            snp, maf, res, lrts = gene_data
            log_env_writer.writerow([episode, step, gene_idx, snp.detach().cpu().numpy(), maf.detach().cpu().numpy(), res.detach().cpu().numpy(), lrts.detach().cpu().numpy()])
        log_env_writer.writerow(["--------------------------------------------------------------"])

#########################################PLOT

params = {'legend.fontsize': 48,
        'figure.figsize': (54, 32),
        'axes.labelsize': 60,
        'axes.titlesize':60,
        'xtick.labelsize':60,
        'ytick.labelsize':60,
        'lines.linewidth': 10}

plt.rcParams.update(params)

def plot_rewards(losses1, losses2, losses3, i_episode=0, path=None, sim=False):
    epochs = range(1, len(losses1) + 1) 

    fig = plt.figure()


    plt.plot(epochs, losses1, 'r', label='Total Rewards')
    plt.plot(epochs, losses2, 'g', label='Utility Rewards')
    plt.plot(epochs, losses3, 'b', label='Privacy rewards')

    plt.title('Rewards of model')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()

    
    if path:
        plt.savefig(f"{path}/rewards/rewards{i_episode}.png")
    # plt.show()
    plt.close()

    if sim:
        plt.grid(True)

        canvas = FigureCanvas(fig)
        canvas.draw()

        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()

        size = canvas.get_width_height()

    # return pygame.image.fromstring(raw_data, size, "RGB")



def plot_individual_rewards(losses1, losses2, losses3, i_episode, path):
    epochs = range(1, len(losses1) + 1)  # Assuming all lists have the same length

    plt.figure()

    plt.subplot(3, 1, 1)
    plt.plot(epochs, losses1, 'r')
    plt.title('Total Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')

    plt.subplot(3, 1, 2)
    plt.plot(epochs, losses2, 'g')
    plt.title('Utility Loss')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')

    plt.subplot(3, 1, 3)
    plt.plot(epochs, losses3, 'b')
    plt.title('Privacy Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')

    plt.savefig(f"{path}/indrewards/indreward{i_episode}.png")
    plt.close()


def plot_lists(values, path, name, episode, thresh=None, xlabel='Episodes', ylabel='Values'):
    fig, ax = plt.figure(), plt.gca()
    ax.plot(np.arange(len(values)), values, label='Values', linewidth=2)

    if thresh is not None:
        ax.axhline(y=thresh, color='r', linestyle='--', label='Threshold', linewidth=1.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(f"{path}/{name}_{episode}.png")
    # plt.show()
    plt.close(fig)

    # plt.grid(True)

    # canvas = FigureCanvas(fig)
    # canvas.draw()

    # renderer = canvas.get_renderer()
    # raw_data = renderer.tostring_rgb()

    # size = canvas.get_width_height()

    # return pygame.image.fromstring(raw_data, size, "RGB")


def plot_two_lists(list1, list2, path, name, episode, thresh=None, label1 = 'Beacon Action', label2='Attacker Action', xlabel='Episodes', ylabel='Values'):
    fig, ax = plt.figure(), plt.gca()
    ax.plot(np.arange(len(list1)), list1, label=label1, linewidth=2)
    ax.plot(np.arange(len(list2)), list2, label=label2, linewidth=2)

    if thresh is not None:
        ax.axhline(y=thresh, color='r', linestyle='--', label='Threshold', linewidth=1.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(f"{path}/{name}_{episode}.png")
    # plt.show()
    plt.close(fig)


def plot_three_lists(list1, list2, list3, path, name, episode, thresh=None, label1 = 'Beacon Action', label2='Attacker Action', label3='Total', xlabel='Episodes', ylabel='Values'):
    fig, ax = plt.figure(), plt.gca()
    ax.plot(np.arange(len(list1)), list1, label=label1, linewidth=2)
    ax.plot(np.arange(len(list2)), list2, label=label2, linewidth=2)
    ax.plot(np.arange(len(list3)), list3, label=label3, linewidth=2)


    if thresh is not None:
        ax.axhline(y=thresh, color='r', linestyle='--', label='Threshold', linewidth=1.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(f"{path}/{name}_{episode}.png")
    # plt.show()
    plt.close(fig)


def plot_comparisons(rewards_list, labels_list):
    # Define a consistent color palette
    colors = sns.color_palette("muted", n_colors=len(rewards_list))

    # Line Plot
    plt.figure()
    for i, rewards in enumerate(rewards_list):
        plt.plot(rewards, marker='o', linestyle='-', color=colors[i], label=labels_list[i])
    plt.xlabel('Victim')
    plt.ylabel('Cumulative Reward Value')
    plt.title('Comparison of Reward Trends')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Bar Plot
    x = np.arange(len(rewards_list[0]))  # Create an array with the length of the lists
    width = 0.8 / len(rewards_list)  # Adjust width based on number of reward lists

    plt.figure()
    for i, rewards in enumerate(rewards_list):
        plt.bar(x + i * width - width * len(rewards_list) / 2, rewards, width, label=labels_list[i], color=colors[i], alpha=0.7)
    plt.xlabel('Victim')
    plt.ylabel('Cumulative Reward Value')
    plt.title('Comparison of Rewards at Each Index')
    plt.xticks(x)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    # Box Plot
    plt.figure()
    plt.boxplot(rewards_list, labels=labels_list, patch_artist=True, medianprops=dict(color="black"))
    for patch, color in zip(plt.gca().artists, colors):
        patch.set_facecolor(color)
    plt.ylabel('Cumulative Reward Value')
    plt.title('Distribution of Rewards')
    plt.grid(True)
    # plt.xticks(rotation=45) 
    plt.tight_layout()
    plt.show()



def violin_plot(data, edge_color, fill_color, positions, ax):
    parts = ax.violinplot(data, positions=positions, 
                          widths=0.5, showmedians=True)
    
    for pc in parts['bodies']:
        pc.set_facecolor(fill_color)
        pc.set_edgecolor(edge_color)
        pc.set_linewidth(1)
        pc.set_alpha(1)
    
    # Customizing the median line
    if 'cmedians' in parts:
        parts['cmedians'].set_color(edge_color)
        parts['cmedians'].set_linewidth(1)
    
    # Customizing the whiskers
    for partname in ('cbars', 'cmins', 'cmaxes'):
        vp = parts[partname]
        vp.set_edgecolor(edge_color)
        vp.set_linewidth(1)


def box_plot(data, edge_color, fill_color, positions, ax):
    """
    Plots a box plot with custom style for each data set.
    
    Parameters:
    data (array): 1D array of data points to plot.
    edge_color (str): Color of the box edges.
    fill_color (str): Fill color of the boxes.
    positions (array): Positions where to place the boxes on the x-axis.
    ax (matplotlib.axes.Axes): The axes to plot on.
    """
    bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True,
                    boxprops=dict(facecolor=fill_color, color=edge_color, linewidth=1),
                    medianprops=dict(color=edge_color, linewidth=1),
                    whiskerprops=dict(color=edge_color, linewidth=1),
                    capprops=dict(color=edge_color, linewidth=1),
                    flierprops=dict(marker='o', color=edge_color, alpha=0.5))

def plot_boxplot(data, labels, title, ylabel):
    """
    Generates a box plot for a given 3D numpy array with custom style and colors.
    
    Parameters:
    data (numpy array): A 3D numpy array with shape (classes, samples, stages).
    """
    color_palette = ["#65879F", "#8B8C89", "#425062", "#8F5C5C", "#CFACAC"]
    yticks = ['100', '200', '300', '400', '500', '600', '700', '800', '900', '1000']
    classes, samples, stages = data.shape
    
    fig, ax = plt.subplots()
    
    x_positions = np.arange(stages) * (classes + 1)
    for class_idx in range(classes):
        class_data = data[class_idx]  # shape (samples, stages)
        
        for stage in range(stages):
            stage_data = class_data[:, stage]
            positions = x_positions + class_idx
            edge_color = 'black'
            fill_color = color_palette[class_idx]
            box_plot(stage_data, edge_color, fill_color, [positions[stage]], ax)
    
    ax.set_xlabel('Query')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Setting the legend
    handles = [mpatches.Rectangle((0, 0), 1, 1, color=color_palette[i], edgecolor='black', linewidth=4) for i in range(classes)]
    ax.legend(handles, labels, loc="lower right")
    
    ax.set_xticks(x_positions + (classes - 1) / 2)
    ax.set_xticklabels([f'{i}' for i in yticks])
    ax.set_xlim(min(x_positions) - 1, max(x_positions) + (classes - 1) + 1)
    
    # Adding borders to separate classes
    for i in range(1, stages):
        ax.axvline(x=(i * (classes + 1)) - 1, color='black', linewidth=2, linestyle='--')

    plt.show()


def plot_violinplot(data, labels, title, ylabel):
    """
    Generates a violin plot for a given 3D numpy array with custom style and colors.
    
    Parameters:
    data (numpy array): A 3D numpy array with shape (classes, samples, stages).
    """
    color_palette = ["#65879F", "#8B8C89", "#425062", "#8F5C5C", "#CFACAC"]
    yticks = [1, 10, 20, 30, 40, 50]
    classes, samples, stages = data.shape
    
    fig, ax = plt.subplots()
    
    x_positions = np.arange(stages) * (classes + 1)
    for class_idx in range(classes):
        class_data = data[class_idx]  # shape (samples, stages)
        
        for stage in range(stages):
            stage_data = class_data[:, stage]
            positions = x_positions + class_idx
            edge_color = 'black'
            fill_color = color_palette[class_idx]
            violin_plot(stage_data, edge_color, fill_color, [positions[stage]], ax)
    
    ax.set_xlabel('Query')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Setting the legend
    handles = [mpatches.Rectangle((0, 0), 1, 1, color=color_palette[i], edgecolor='black', linewidth=4) for i in range(classes)]
    ax.legend(handles, labels, loc="lower right")
    
    ax.set_xticks(x_positions + (classes - 1) / 2)
    ax.set_xticklabels([f'{i}' for i in yticks])
    ax.set_xlim(min(x_positions) - 1, max(x_positions) + (classes - 1) + 1)
    
    # Adding borders to separate classes
    for i in range(1, stages):
        ax.axvline(x=(i * (classes + 1)) - 1, color='black', linewidth=2, linestyle='--')

    plt.show()


params = {'legend.fontsize': 25,
        'figure.figsize': (25, 10),  # Reduced figure size
        'axes.labelsize': 40,
        'axes.titlesize': 50,
        'xtick.labelsize': 35,
        'ytick.labelsize': 30,
        'lines.linewidth': 8}  # Adjusted line width

plt.rcParams.update(params)
def line_plot(line_data, labels, title, ylabel_line):
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))  # Single subplot now

    color_palette = ["#2E2E2E", "#E74C3C", "#3498DB", "#1ABC9C", "#E67E22", "#F1C40F", "#65879F", "#8B8C89", "#425062", "#8F5C5C", "#CFACAC"]
    x_ticks = ['1', '100', '200', '300', '400', '500', '600', '700', '800', '900', '1000']
    classes, stages = line_data.shape
    
    x_positions = np.arange(stages)
    
    # Plotting the line data
    for class_idx in range(classes):
        class_data = line_data[class_idx, :]  # shape (stages,)
        edge_color = 'black'
        line_color = color_palette[class_idx]
        
        # Plotting the line with markers
        ax1.plot(x_positions, class_data, marker='o', color=line_color, markeredgewidth=1, markeredgecolor=edge_color, linewidth=3)

    ax1.set_ylabel(ylabel_line, fontsize=15)
    ax1.set_ylim(0, 1)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(x_ticks, fontsize=20)
    ax1.spines['bottom'].set_visible(True)  # Bottom border is visible for a single plot

    # y_ticks = ax1.get_yticks()
    # ax1.set_yticklabels(['' if tick == 0 else f'{tick:.1f}' for tick in y_ticks])
    ax1.tick_params(axis='y', labelsize=12)
    ax1.set_xlabel('Query Number', fontsize=20)
    
    # Adding legend for line plot
    handles = [mpatches.Patch(facecolor=color_palette[i], label=labels[i], edgecolor='black', linewidth=1) for i in range(classes)]
    ax1.legend(handles, labels, loc="lower left", fontsize=12)
    
    # Adjust subplot layout
    plt.tight_layout()
    plt.show()

def scatter_plot(y_data, size_data, labels, title, ylabel):
    fig, ax = plt.subplots()
    color_palette = ["#65879F", "#8B8C89", "#425062", "#8F5C5C", "#CFACAC"]
    x_ticks = ['100', '200', '300', '400', '500', '600', '700', '800', '900', '1000']
    classes, samples = y_data.shape

    x_positions = np.arange(samples)

    # Define marker shape groups based on size ranges
    size_ranges = [(1, 10), (10, 15), (10, 20), (20, 30)]
    marker_styles = ['v', '^', 's', 'o']  # Corresponding marker shapes for the ranges
    size_labels = ["1-5", "5-10", "10-15", "15-20"]  # Labels for legend

    # Loop through each class
    for class_idx in range(classes):
        y_class_data = y_data[class_idx, :]  # y-axis data for the class
        size_class_data = size_data[class_idx, :]  # Size data for the class
        edge_color = 'black'
        marker_color = color_palette[class_idx]

        # Scatter plot with custom marker shapes based on size ranges
        for i, size_range in enumerate(size_ranges):
            mask = (size_class_data >= size_range[0]) & (size_class_data < size_range[1])
            ax.scatter(x_positions[mask], y_class_data[mask], color=marker_color, edgecolor=edge_color,
                       marker=marker_styles[i], linewidth=1.5, s=1000)

        ax.plot(x_positions, y_class_data, color=marker_color, linewidth=5)

    ax.set_xlabel('Query')
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Legend for the marker shapes
    shape_handles = [plt.Line2D([], [], color='w', marker=marker_style, markersize=30,
                                markerfacecolor='gray', label=label, markeredgecolor='black')
                     for marker_style, label in zip(marker_styles, size_labels)]
    legend1 = ax.legend(shape_handles, size_labels, title="Num of Predicted", loc="lower left", title_fontsize=36)
    ax.add_artist(legend1)

    # Setting the class labels legend
    class_handles = [Line2D([0], [0], color=color_palette[i], linewidth=6, label=labels[i]) for i in range(classes)]
    ax.legend(class_handles, labels, loc="center right")

    # Set x-ticks and show only selected x-ticks for clarity
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_ticks)

    plt.show()

params = {'legend.fontsize': 25,
        'figure.figsize': (20, 10),  # Reduced figure size
        'axes.labelsize': 25,
        'axes.titlesize': 50,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'lines.linewidth': 8}  # Adjusted line width

plt.rcParams.update(params)

def line_and_bar_plot(line_data, bar_data, labels, title, ylabel_line, ylabel_bar):
    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    fig, ax1 = plt.subplots(1, 1)

    
    color_palette = ["#2E2E2E", "#E74C3C", "#3498DB", "#1ABC9C", "#E67E22", "#F1C40F", "#65879F", "#8B8C89", "#425062", "#8F5C5C", "#CFACAC"]
    x_ticks = ['1', '100', '200', '300', '400', '500', '600', '700', '800', '900', '1000']
    classes, stages = line_data.shape
    
    x_positions = np.arange(stages)
    bar_width = 0.12  # Adjusted Bar width for better clarity
    
    # Plotting the line data
    for class_idx in range(classes):
        class_data = line_data[class_idx, :]  # shape (stages,)
        edge_color = 'black'
        line_color = color_palette[class_idx]
        
        # Plotting the line with markers
        ax1.plot(x_positions, class_data, marker='o', color=line_color, markeredgewidth=2, markeredgecolor=edge_color, linewidth=5, alpha=0.8)

    ax1.set_ylabel(ylabel_line)
    ax1.set_ylim(0.9, 1)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(x_ticks)
    # ax1.spines['bottom'].set_visible(False)

    # y_ticks = ax1.get_yticks()
    # ax1.set_yticklabels(['' if tick == 0 else f'{tick:.1f}' for tick in y_ticks])
    ax1.tick_params(axis='y', labelsize=20)
    
    # Adding legend for line plot
    handles = [mpatches.Patch(facecolor=color_palette[i], label=labels[i], edgecolor='black', linewidth=3, alpha=0.8) for i in range(classes)]
    ax1.legend(handles, labels, loc="lower left", fontsize=20)
    
    # # Plotting the bar data on the lower axis
    # for class_idx in range(classes):
    #     bar_vals = bar_data[class_idx, :]  # shape (stages,)
    #     bar_color = color_palette[class_idx]
        
    #     # Offset the bar positions to avoid overlap
    #     bars = ax2.bar(x_positions + (class_idx - classes / 2) * bar_width, bar_vals + 0.5, width=bar_width, color=bar_color, alpha=0.8)
        
    #     # Annotate bars with their values
    #     for bar in bars:
    #         height = bar.get_height()
    #         ax2.annotate(f'{height-0.5:.0f}',  # format the label
    #                      xy=(bar.get_x() + bar.get_width() / 2, height),  # set the label position
    #                      xytext=(0, 3),  # 3 points vertical offset
    #                      textcoords="offset points",
    #                      ha='center', va='bottom',
    #                      fontsize=12)

    # # Set ylabel for bar plot
    # ax2.set_ylabel(ylabel_bar, fontsize=30)
    ax1.set_xlabel('Query Number')
    
    # ax2.set_xticks(x_positions)
    # ax2.set_xticklabels(x_ticks)
    # ax2.tick_params(axis='y', labelsize=25)
    # ax2.spines['top'].set_visible(False)

    # Adjust subplot spacing
    plt.subplots_adjust(top=0.95, bottom=0.08, left=0.07, right=0.97, hspace=0.15)  # Fine-tuned margins
    plt.tight_layout()
    # plt.show()


def line_and_bar_plot2(line_data, bar_data, labels, title, ylabel_line, ylabel_bar):
    fig, ax1 = plt.subplots()
    
    color_palette = ["#2E2E2E", "#E74C3C", "#3498DB", "#1ABC9C", "#E67E22", "#F1C40F", "#65879F", "#8B8C89", "#425062", "#8F5C5C", "#CFACAC"]
    x_ticks = ['1', '100', '200', '300', '400', '500', '600', '700', '800', '900', '1000']
    classes, stages = line_data.shape
    
    x_positions = np.arange(stages)
    bar_width = 0.12  # Adjusted Bar width for better clarity
    
    # Plotting the line data
    for class_idx in range(classes):
        class_data = line_data[class_idx, :]  # shape (stages,)
        edge_color = 'black'
        line_color = color_palette[class_idx]
        
        # Plotting the line with markers
        ax1.plot(x_positions, class_data, marker='o', color=line_color, markeredgewidth=2, markeredgecolor=edge_color, linewidth=3)

    ax1.set_ylabel(ylabel_line)
    ax1.set_ylim(0, 1)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(x_ticks)
    ax1.set_yticks(np.linspace(0, 1, 6))
    
    # Adding legend for line plot
    handles = [mpatches.Patch(facecolor=color_palette[i], label=labels[i], edgecolor='black', linewidth=2) for i in range(classes)]
    ax1.legend(handles, labels, loc="center left")
    
    # Create a twin axis to plot the bar chart
    ax2 = ax1.twinx()
    
    # Plotting the bar data on the same axis
    for class_idx in range(classes):
        bar_vals = bar_data[class_idx, :]  # shape (stages,)
        bar_color = color_palette[class_idx]
        
        # Offset the bar positions to avoid overlap
        bars = ax2.bar(x_positions + (class_idx - classes / 2) * bar_width, bar_vals+0.5, width=bar_width, color=bar_color, alpha=0.7)
        
        # Annotate bars with their values
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.0f}',  # format the label
                         xy=(bar.get_x() + bar.get_width() / 2, height),  # set the label position
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom',
                         fontsize=12)

    # Set ylabel for bar plot
    ax2.set_ylabel(ylabel_bar)
    ax2.set_ylim(0, 100)
    ax2.set_yticks([0, 10, 20, 30])
    ax2.tick_params(axis='y', labelsize=30)

    
    ax1.set_xlabel('Query Number')
    
    # Adjust subplot spacing
    # plt.title(title, fontsize=18)
    plt.tight_layout()
    plt.show()
    
def line_and_bar_plot3(line_data, bar_data, labels, ylabel_line, ylabel_bar):
    # line_data and bar_data should have shapes (4, 7, 11), so we iterate over the first dimension
    num_groups, classes, stages = line_data.shape
    
    fig, axs = plt.subplots(2, 2)  # Create a 2x2 grid for the panels
    color_palette = ["#2E2E2E", "#E74C3C", "#3498DB", "#1ABC9C", "#E67E22", "#F1C40F", "#65879F", "#8B8C89", "#425062", "#8F5C5C", "#CFACAC"]
    x_ticks = ['1', '100', '200', '300', '400', '500', '600', '700', '800', '900', '1000']
    x_positions = np.arange(stages)
    bar_width = 0.12  # Adjusted Bar width for better clarity


    labels_for_plots = ['A', 'B', 'C', 'D']  # To mark each plot

    # Iterate over the first dimension (groups) and plot each group in a separate panel
    for group_idx in range(num_groups):
        ax = axs[group_idx // 2, group_idx % 2]  # Get the correct panel (axes) for the 2x2 grid
        ax.set_xticks([])  # Removes the x-ticks
        ax.set_xticklabels([])  # Removes the x-tick labels
        ax.set_yticks([])  # Removes the x-ticks
        ax.set_yticklabels([])  # Removes the x-tick labels

        ax.text(0, 1.1, labels_for_plots[group_idx], transform=ax.transAxes, 
                fontsize=25, fontweight='bold', va='top', ha='right')

        if group_idx != num_groups - 1: 
            # Create two subplots (ax1 for line plot, ax2 for bar plot) in the current panel
            ax1 = ax.inset_axes([0, 0.28, 1, 0.7])  # Upper inset for the line plot
            ax2 = ax.inset_axes([0, 0, 1, 0.25])  # Lower inset for the bar plot
            ax1.set_xticks([])  # Removes the x-ticks
            ax1.set_xticklabels([])  # Removes the x-tick labels
            ax1.spines['bottom'].set_visible(False)
            ax1.spines['top'].set_visible(False)


        else:
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax1 = ax.inset_axes([0, 0.28, 1, 0.72])
            ax1.set_xlabel('Query Number', fontsize=20)
            ax1.set_xticks(x_positions)
            ax1.set_xticklabels(x_ticks)
            ax1.tick_params(axis='x', labelsize=25)



        # Plotting the line data on the upper axis (ax1)
        for class_idx in range(classes):
            class_data = line_data[group_idx, class_idx, :]  # shape (stages,)
            edge_color = 'black'
            line_color = color_palette[class_idx]

            ax1.plot(x_positions, class_data, marker='.', color=line_color, markeredgewidth=2, markeredgecolor=edge_color, linewidth=5, alpha=0.8)

        ax1.set_ylabel(ylabel_line, fontsize=20)
        ax1.set_ylim(0, None)
        # ax1.set_xticks(x_positions)
        # ax1.set_xticklabels(x_ticks)

        y_ticks = np.linspace(0, 1, 5)
        ax1.set_yticks(y_ticks)
        ax1.set_yticklabels([f'{tick:.1f}' if tick != 0 else '' for tick in y_ticks])
        ax1.tick_params(axis='y', labelsize=25)

        

        # Adding legend for line plot
        handles = [mpatches.Patch(facecolor=color_palette[i], label=labels[i], edgecolor='black', linewidth=2, alpha=0.8) for i in range(classes)]
        # ax1.legend(handles, labels, loc="lower left", fontsize=15)

        # Plotting the bar data on the lower axis (ax2)
        if group_idx != num_groups - 1: 
            for class_idx in range(classes):
                bar_vals = bar_data[group_idx, class_idx, :]  # shape (stages,)
                bar_color = color_palette[class_idx]

                bars = ax2.bar(x_positions + (class_idx - classes / 2) * bar_width, bar_vals + 0.5, width=bar_width, color=bar_color, alpha=0.8)

                # # Annotate bars with their values
                # for bar in bars:
                #     height = bar.get_height()
                #     ax2.annotate(f'{height-0.5:.0f}', 
                #                 xy=(bar.get_x() + bar.get_width() / 2, height),
                #                 xytext=(0, 3), 
                #                 textcoords="offset points",
                #                 ha='center', va='bottom',
                #                 fontsize=10)

            ax2.set_ylabel(ylabel_bar, fontsize=20)
            ax2.set_xlabel('Query Number', fontsize=20)
            ax2.set_xticks(x_positions)
            ax2.set_xticklabels(x_ticks)
            ax2.set_yticks([0, 10, 20, 30])

            ax2.tick_params(axis='y', labelsize=20)
            ax2.tick_params(axis='x', labelsize=25)
            ax2.spines['top'].set_visible(False)
            ax2.grid(True, linestyle='--', color='gray', linewidth=0.7, alpha=0.6)

    # fig.legend(handles=handles, loc='lower right', ncol=classes, fontsize=23, frameon=False)
    # handles = [mpatches.Patch(facecolor=color_palette[i], label=labels[i], edgecolor='black', linewidth=2, alpha=0.5) for i in range(classes)]
    # fig.legend(handles, labels, loc="lower right", fontsize=15)
    legend = fig.legend(handles=handles, loc='lower right', bbox_to_anchor=(0.98, 0.03), ncol=4, fontsize=20)  

    plt.tight_layout()
    plt.show()