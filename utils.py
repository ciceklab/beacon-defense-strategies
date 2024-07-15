import matplotlib.pyplot as plt
import numpy as np
# import pygame
from matplotlib_inline.backend_inline import FigureCanvas
import csv

import torch


#######################################LRT
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

# params = {'legend.fontsize': 48,
#         'figure.figsize': (54, 32),
#         'axes.labelsize': 60,
#         'axes.titlesize':60,
#         'xtick.labelsize':60,
#         'ytick.labelsize':60,
#         'lines.linewidth': 10}

# plt.rcParams.update(params)

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