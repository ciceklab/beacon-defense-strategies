import numpy as np
import copy
import os
import csv
import math
import random

import torch

from utils import calculate_ind_lrt

class Attacker():
    def __init__(self, args, victim, control, mafs):
        self.args = args
        self.mafs = mafs
        self.victim = victim
        self.attacker_control = control
        # print("Initializing {} Attacker".format(self.args.attacker_type))


        ######################## Init the Beacon info
        temp_maf = torch.Tensor(self.mafs)
        responses = torch.ones(size=(self.args.gene_size,))*-1
        current_query = torch.zeros(size=(self.args.gene_size,))
        lrts = torch.zeros(size=(self.args.gene_size,))
        self.victim_info = torch.stack([victim, temp_maf, responses, lrts], dim=-1)

        self.victim = victim

        # print("Victim", self.victim_info)

        ######################## Init the Control info
        temp_maf = torch.Tensor(self.mafs).unsqueeze(0).expand(self.args.a_control_size, -1)
        responses = torch.ones(size=(self.args.a_control_size, self.args.gene_size))*-1
        current_query = torch.ones(size=(self.args.a_control_size, self.args.gene_size))
        lrts = torch.zeros(size=(self.args.a_control_size, self.args.gene_size))
        self.control_info = torch.stack([torch.Tensor(self.attacker_control), temp_maf, responses, current_query, lrts], dim=-1)


        #########################
        # if self.args.attacker_type == "optimal":
        self.optimal_queries = self._init_optimal_attacker()

        ######################## Initializing the control LRTS
        self.control_lrts = torch.zeros(size=(self.args.a_control_size,))
        self.victim_lrt = torch.as_tensor(0)


    #################################################################################################
    # Get Beacon state
    def get_state(self)->float:
        # print("Attacker Side: ")

        min_control_lrt = torch.min(self.control_lrts)
        max_control_lrt = torch.max(self.control_lrts)

        # print("\tControls min: {} and max {} LRT".format(min_control_lrt, max_control_lrt))
        # print("\tVictim: {} LRT".format(self.victim_lrt))

        temp_maf = torch.Tensor(self.mafs)
        # temp_maf = torch.cat((temp_maf, min_control_lrt.unsqueeze(0), max_control_lrt.unsqueeze(0), self.victim_lrt.unsqueeze(0)))

        # victim = torch.cat((self.victim, torch.tensor([2, 2, 2])))

        state = torch.stack([self.victim, temp_maf], dim=-1)

        # print("temp_maf: ", temp_maf)
        # print("updated_victim: ", updated_victim)

        # print("state: ", state.size())

        victim_size = torch.nonzero(self.victim == 1).size(0) 
        # print("victim_size", victim_size)

        # From victim info get the SNPs that have not queried
        spns_msk = (self.victim_info[:, 0] == 1) & (self.victim_info[:, 2] == -1)  # SNPs
        filtered_ind = self.victim_info[spns_msk]
        min_maf = min(filtered_ind[:, 1])
        max_maf = max(filtered_ind[:, 1])
        # print(filtered_ind[:, 1])
        mean_maf = torch.mean(filtered_ind[:, 1])


        # return [min_control_lrt, self.victim_lrt, victim_size, min_maf, max_maf, mean_maf]
        return self.victim_info

    #################################################################################################
    # Update Beacon
    def update(self, attacker_action):
        self.attacker_action = attacker_action
        self.victim_info[self.attacker_action, 2] = torch.as_tensor(1) #TRUTH
        # print(self.victim_info)


        # Update the Attacker control group and calculate the LRT values
        victim_lrts = calculate_ind_lrt(self.victim_info, gene_size=self.args.gene_size, number_of_people=self.args.beacon_size)
        self.victim_info[:, 3] = torch.Tensor(victim_lrts)
        self.victim_lrt = torch.sum(victim_lrts)

        #Update Control Group
        self.control_info[:, self.attacker_action, 2] = torch.as_tensor(1) #TRUTH
        self.control_lrts = self._calc_group_lrts(self.control_info, True)


    #################################################################################################
    # Reward Calculation
    def calc_reward(self, beacon_action, step)->float:
        done=False
        pvalue = self._calc_pvalue()

        # preward1 = -pvalue
        # preward2 = (10 - step) / 10
        preward1 = -1

        total = preward1 #+ preward2

        if pvalue < 0.05:
            total += 40
            done=True
            print("⛔⛔⛔ Attacker Identified the VICTIM ⛔⛔⛔")

        # print("Attacker Rewards: \n\t1-pvalue: {}\n\tutility: {}\n\ttotal: {}".format(preward1, preward1, total))
        # print("-----------------------------------------------------------------")

        return total, done, [preward1, self.victim_lrt]
    

    #################################################################################################
    # Attacker Action
    def act(self, current_step, agent_action=None):
        if self.args.attacker_type == "optimal":
            return self.optimal_queries[current_step]

        if self.args.attacker_type == "agent":
            return self.optimal_queries[agent_action]
        


    #################################################################################################
    #LRT PVALUES
    def _calc_group_lrts(self, group, save=True)->float:
        lrt_values = []
        for index, individual in enumerate(group):
            lrts = calculate_ind_lrt(group[index, :, :], gene_size=self.args.gene_size, number_of_people=self.args.beacon_size)


            lrt_values.append(torch.sum(lrts))############ Sum of the LRTs
            if save:
                group[index, :, 4] = torch.Tensor(lrts)

        return torch.Tensor(lrt_values)
    
    def _calc_pvalue(self):
        victim_lrt = copy.deepcopy(self.victim_lrt)
        control_lrts = copy.deepcopy(self.control_lrts)

        pvalue=torch.sum(victim_lrt >= control_lrts) / self.args.a_control_size
        return torch.Tensor(pvalue)


    #################################################################################################
    # Optimal Attacker
    def _init_optimal_attacker(self):
        # print("\tInitializing the Optimal Attack Strategy")
        maf_values = self.mafs
        sorted_gene_indices = torch.argsort(maf_values)  # Sort the MAFs using PyTorch
        # print("Victim : {}".format(self.victim))
        # print("Victim Sorted Genes: {}".format(sorted_gene_indices))
        
        mask_one = (self.victim[sorted_gene_indices] == 1)  # Get the SNPs with value 1
        mask_zero = (self.victim[sorted_gene_indices] == 0)  # Get the SNPs with value 0
        
        # Use torch.cat to concatenate the indices
        queries = torch.cat((sorted_gene_indices[mask_one], sorted_gene_indices[mask_zero]))#[:self.args.max_queries]
        
        # print(f"\tAttacker Queries: {queries}")
        return queries
    




