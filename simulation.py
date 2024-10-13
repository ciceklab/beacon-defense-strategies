import pandas as pd
import numpy as np
import pickle
import argparse
import os
import random
import torch

from utils import *

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
    parser.add_argument('--episodes', default=19, type=int, metavar='N', help='Number of episodes for training agent.')
    parser.add_argument('--seed', default=19, type=int, help='Seed for reproducibility')
    parser.add_argument('--evaluate', default=True, type=bool, help='Evaluation or Not')


    # Env Properties
    parser.add_argument('--a_control_size', default=50, type=int, help='Attack Control group size')
    parser.add_argument('--b_control_size', default=50, type=int, help='Beacon Control group size')
    parser.add_argument('--gene_size', default=100000, type=int, help='States gene size')
    parser.add_argument('--beacon_size', default=20, type=int, help='Beacon population size')
    parser.add_argument('--victim_prob', default=1, type=float, help='Victim inside beacon or not!')
    parser.add_argument('--max_queries', default=1000, type=int, help='Maximum queries per episode')


    parser.add_argument('--attacker_type', default="optimal", choices=["random", "optimal", "agent"], type=str, help='Type of the attacker')
    parser.add_argument('--beacon_type', default="truth", choices=["urandom", "random", "agent", "truth", "beacon_strategy"], type=str, help='Type of the beacon')

    parser.add_argument('--beacon_agent', default="td", choices=["td", "ppo"], type=str, help='Type of the beacon')

    parser.add_argument('--pop_reset_freq', default=100000000, type=int, help='Reset Population Frequency (Epochs)')
    parser.add_argument('--plot-freq', default=1, type=int, metavar='N', help='Plot Frequencies')

    # utils
    parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--results-dir', default='./results/simulation', type=str, metavar='PATH', help='path to cache (default: none)')

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

import os
import joblib

print("Reading DATA")
# Cache file path
cache_path = "/data6/sobhan/Beacons/dataset/binary_cache.joblib"

# Check if the cached file exists
if os.path.exists(cache_path):
    print("Exists")
    # Load from cache
    binary = joblib.load(cache_path)
else:
    # If cache doesn't exist, process and save to cache
    beacon = pd.read_csv(os.path.join("/mnt/kerem/CEU", "Beacon_164.txt"), index_col=0, delim_whitespace=True)
    reference = pickle.load(open(os.path.join("/mnt/kerem/CEU", "reference.pickle"), "rb"))
    binary = np.logical_and(beacon.values != reference, beacon.values != "NN").astype(int)
    
    # Save the processed binary data to cache for future use
    joblib.dump(binary, cache_path)

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
print(binary.shape) #(164, 4029840)

from ppo import PPO
from td import TD3


from env import Env

args=args_create()


def simulate(args, beacon_type, attacker_type, attacker_resume=None, beacon_resume=None):
    args.beacon_type = beacon_type 
    args.attacker_type = attacker_type 
    # args.beacon_agent = beacon_agent 
    env = Env(args, maf_values, binary)
    ################ PPO hyperparameters ################
    K_epochs = 300         # update policy for K epochs
    eps_clip = 0.1           # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0001      # learning rate for actor network
    lr_critic = 0.0001        # learning rate for critic network

    i_episode = 0

    attacker_state_dim = 400
    attacker_action_dim = 50

    beacon_rewards = []
    attacker_rewards = []
    privacies = []
    utilities = []

    if attacker_type == "agent":
        attacker_agent = PPO(attacker_state_dim, attacker_action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, False, None)
        attacker_agent.load(attacker_resume)

    if beacon_type == "agent":
        state_dim = 18
        action_dim = 1
        beacon_agent = TD3(state_dim, action_dim, max_action=1)
        beacon_agent.load(beacon_resume)

    while i_episode < args.episodes:
        beacon_rewardss = []
        attacker_rewardss = []
        privaciess = []
        utilitiess = []
        for t in range(1, args.max_queries+1):
            if attacker_type == "agent" and beacon_type == "agent":
                beacon_state, rewards, done, pu  = env.step(attacker_agent=attacker_agent, beacon_agent=beacon_agent)
            elif attacker_type == "agent":
                beacon_state, rewards, done, pu  = env.step(attacker_agent=attacker_agent)
            elif beacon_type == "agent":
                beacon_state, rewards, done, pu  = env.step(beacon_agent=beacon_agent)
            else:
                beacon_state, rewards, done, pu  = env.step()

            # print(beacon_state[0])

            beacon_rewardss.append(rewards[0])
            attacker_rewardss.append(rewards[1])
            privaciess.append(pu[0])
            utilitiess.append(pu[1])
            if done:
                break
        
        beacon_rewards.append(beacon_rewardss)
        attacker_rewards.append(attacker_rewardss)
        privacies.append(privaciess)
        utilities.append(utilitiess)

        print("Victim: {} \t Current Episode Reward : {}".format(env.victim_id, rewards[1]))
        env.reset()
        i_episode += 1
    return beacon_rewards, attacker_rewards, privacies, utilities

evaluations = [
    # {
    #     "beacon_type": "baseline",
    #     "attacker_type": "optimal",
    #     "beacon_resume": None,
    #     "attacker_resume": None
    # },
    {
        "beacon_type": "baseline",
        "attacker_type": "optimal",
        "beacon_resume": "/data6/sobhan/Beacons/results/train/run34/weights",
        "attacker_resume": None
    },
    # {
    #     "beacon_type": "random",
    #     "attacker_type": "random",
    #     "beacon_resume": None,
    #     "attacker_resume": None
    # }
]

beacon_rewards=[]
attacker_rewards=[]
privacies=[]
utilities=[]

for eval in evaluations:
    print(eval)
    res = simulate(args, beacon_type=eval["beacon_type"], attacker_type=eval["attacker_type"], beacon_resume=eval['beacon_resume'], attacker_resume=eval["attacker_resume"])
    beacon_rewards.append(res[0])
    attacker_rewards.append(res[1])
    privacies.append(res[2])
    utilities.append(res[3])