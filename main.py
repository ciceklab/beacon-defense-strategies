import pandas as pd
import numpy as np
import pickle
import argparse
import os
import random

import torch

def reproducibility(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
reproducibility(3)


# set device to cpu or cuda
device = torch.device('cpu')

if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")


def args_create():
    # @title Arguments
    parser = argparse.ArgumentParser(description='Actor Critic')

    parser.add_argument('--data', default="/mnt/kerem/CEU", type=str, help='Dataset Path')
    parser.add_argument('--epochs', default=64, type=int, metavar='N', help='Number of epochs for training agent.')
    parser.add_argument('--episodes', default=10000, type=int, metavar='N', help='Number of episodes for training agent.')
    parser.add_argument('--lr', '--learning-rate', default=0.005, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', default=0.0001, type=float, help='Weight decay for training optimizer')
    parser.add_argument('--seed', default=3, type=int, help='Seed for reproducibility')
    parser.add_argument('--model-name', default="PPO", type=str, help='Model name for saving model.')
    parser.add_argument('--gamma', default=0.99, type=float, metavar='N', help='The discount factor as mentioned in the previous section')

    # Model
    parser.add_argument("--latent1", default=256, required=False, help="Latent Space Size for first layer of network.")
    parser.add_argument("--latent2", default=256, required=False, help="Latent Space Size for second layer of network.")

    # Env Properties
    parser.add_argument('--a_control_size', default=50, type=int, help='Attack Control group size')
    parser.add_argument('--b_control_size', default=50, type=int, help='Beacon Control group size')
    parser.add_argument('--gene_size', default=100, type=int, help='States gene size')
    parser.add_argument('--beacon_size', default=10, type=int, help='Beacon population size')
    parser.add_argument('--victim_prob', default=1, type=float, help='Victim inside beacon or not!')
    parser.add_argument('--max_queries', default=5, type=int, help='Maximum queries per episode')


    parser.add_argument('--attacker_type', default="agent", choices=["random", "optimal", "agent"], type=str, help='Type of the attacker')
    parser.add_argument('--beacon_type', default="truth", choices=["random", "agent", "truth"], type=str, help='Type of the beacon')


    parser.add_argument('--pop_reset_freq', default=100000000, type=int, help='Reset Population Frequency (Epochs)')
    parser.add_argument('--update_freq', default=20, type=int, help='Train Agent model frequency')
    parser.add_argument('--plot-freq', default=10, type=int, metavar='N', help='Plot Frequencies')
    parser.add_argument('--val-freq', default=20, type=int, metavar='N', help='Validation frequencies')
    parser.add_argument('--control-lrts', default=None, type=str, help='Control groups LRTS path')



    parser.add_argument("--state_dim", default=(4,), required=False, help="State Dimension")
    parser.add_argument("--n-actions", default=1, required=False, help="Actions Count for each state")


    # utils
    parser.add_argument('--resume', default="", type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--results-dir', default='./results/train', type=str, metavar='PATH', help='path to cache (default: none)')

    # args = parser.parse_args()  # running in command line
    args = parser.parse_args('')  # running in ipynb

    args.results_dir = os.path.join(args.results_dir, "run"+str(len(os.listdir(args.results_dir))))
    os.makedirs(args.results_dir)
    os.makedirs(args.results_dir+"/logs")
    os.makedirs(args.results_dir+"/rewards")
    os.makedirs(args.results_dir+"/indrewards")
    os.makedirs(args.results_dir+"/actions")
    os.makedirs(args.results_dir+"/pvalues")

    args.device = device

    print(args)
    return args

# args = args_create()

# CEU Beacon - it contains 164 people in total which we will divide into groups to experiment
beacon = pd.read_csv(os.path.join("/mnt/kerem/CEU", "Beacon_164.txt"), index_col=0, delim_whitespace=True)
# Reference genome, i.e. the genome that has no SNPs, all major allele pairs for each position
reference = pickle.load(open(os.path.join("/mnt/kerem/CEU", "reference.pickle"),"rb"))
# Binary representation of the beacon; 0: no SNP (i.e. no mutation) 1: SNP (i.e. mutation)
binary = np.logical_and(beacon.values != reference, beacon.values != "NN").astype(int)

# Table that contains MAF (minor allele frequency) values for each position. 
maf = pd.read_csv(os.path.join("/mnt/kerem/CEU", "MAF.txt"), index_col=0, delim_whitespace=True)
maf.rename(columns = {'referenceAllele':'major', 'referenceAlleleFrequency':'major_freq', 
                      'otherAllele':'minor', 'otherAlleleFrequency':'minor_freq'}, inplace = True)
maf["maf"] = np.round(maf["maf"].values, 3)
# Same variable with sorted maf values
sorted_maf = maf.sort_values(by='maf')
# Extracting column to an array for future use
maf_values = maf["maf"].values

binary = binary.T
binary.shape #(164, 4029840)

has_continuous_action_space = True                

action_std = 0.4             # starting std for action distribution (Multivariate Normal)
action_std_decay_rate = 0.0025       # linearly decay action_std (action_std = action_std - action_std_decay_rate)
min_action_std = 0.05                # minimum action_std (stop decay after action_std <= min_action_std)
action_std_decay_freq = int(2.5e5)

################ PPO hyperparameters ################
K_epochs = 200          # update policy for K epochs
eps_clip = 0.1              # clip parameter for PPO
gamma = 0.99                # discount factor

lr_actor = 0.0001       # learning rate for actor network
lr_critic = 0.0001       # learning rate for critic network


import sys
from env import Env
from ppo import PPO
from engine import train_beacon, train_attacker

args = args_create()
def main():
    env = Env(args, beacon, maf_values, binary)
    
    if args.beacon_type == "agent":
        state_dim = 9
        action_dim = 1
        # initialize a PPO agent
        beacon_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
        train_beacon(args, env, beacon_agent)
    else:
        attacker_state_dim = 400
        attacker_action_dim = 10

        ################ PPO hyperparameters ################
        K_epochs = 300         # update policy for K epochs
        eps_clip = 0.2           # clip parameter for PPO
        gamma = 0.99                # discount factor

        lr_actor = 0.001       # learning rate for actor network
        lr_critic = 0.001        # learning rate for critic network

        attacker_agent = PPO(attacker_state_dim, attacker_action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, False, None)
        train_attacker(args, env, attacker_agent)

        # attacker_agent.buffer.print_buffer()


if __name__ == '__main__':
    # with open(args.results_dir + '/output.txt', 'w') as sys.stdout:
    main()