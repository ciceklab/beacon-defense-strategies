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

        # print("Victim", self.victim_info.size())

        ######################## Init the Control info
        temp_maf = torch.Tensor(self.mafs).unsqueeze(0).expand(self.args.a_control_size, -1)
        responses = torch.ones(size=(self.args.a_control_size, self.args.gene_size))*-1
        current_query = torch.ones(size=(self.args.a_control_size, self.args.gene_size))
        lrts = torch.zeros(size=(self.args.a_control_size, self.args.gene_size))
        self.control_info = torch.stack([torch.Tensor(self.attacker_control), temp_maf, responses, current_query, lrts], dim=-1)


        #########################
        # if self.args.attacker_type == "optimal":
        self.optimal_queries = self._init_optimal_attacker()

        self.agent_queries = self._init_agent_attacker()

        ######################## Initializing the control LRTS
        self.control_lrts = torch.zeros(size=(self.args.a_control_size,))
        self.victim_lrt = torch.as_tensor(0)


    #################################################################################################
    # Get Beacon state
    def get_state(self)->float:
        # print("Attacker Side: ")

        min_control_lrt = torch.min(self.control_lrts)
        mean_control_lrt = torch.mean(self.control_lrts)
        # print(self.agent_queries)

        victim_info = copy.deepcopy(self.victim_info[self.agent_queries, :])

        multiplied_values = victim_info[:, 0] * victim_info[:, 1]
        new_victim = torch.cat([multiplied_values.unsqueeze(1), victim_info[:, 2:]], dim=1)

        flattened_victim = new_victim.flatten()
        final_victim = torch.cat([flattened_victim, min_control_lrt.unsqueeze(0), mean_control_lrt.unsqueeze(0)], dim=0)
        # print("final_victim.size(", final_victim.size())
        return final_victim

    #################################################################################################
    # Update Beacon
    def update(self, beacon_action, attacker_action):
        self.attacker_action = attacker_action
        self.victim_info[self.attacker_action, 2] = torch.as_tensor(beacon_action) #TRUTH
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
        preward2 = beacon_action
        preward1 = -1

        total = preward1 + preward2

        if pvalue < 0.05:
            total += 40
            done=True
            print("⛔⛔⛔ Attacker Identified the VICTIM ⛔⛔⛔")

        # print("Attacker Rewards: \n\t1-pvalue: {}\n\tutility: {}\n\ttotal: {}".format(preward1, preward1, total))
        # print("-----------------------------------------------------------------")

        return total, done, [preward1, preward2]
    

    #################################################################################################
    # Attacker Action
    def act(self, current_step, agent_action=None):
        if self.args.attacker_type == "optimal":
            return self.optimal_queries[current_step]

        if self.args.attacker_type == "agent":
            return self.agent_queries[agent_action]
        
        if self.args.attacker_type == "random":
            return torch.randint(0, self.args.gene_size, (1,)).item()
        


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
    

    def _init_agent_attacker(self):
        # print("\tInitializing the Optimal Attack Strategy")
        maf_values = self.mafs
        sorted_gene_indices = torch.argsort(maf_values)  # Sort the MAFs using PyTorch
        # print("Victim : {}".format(self.victim))
        # print("Victim Sorted Genes: {}".format(sorted_gene_indices))
        
        mask_one = sorted_gene_indices[(self.victim[sorted_gene_indices] == 1)][:1000:2] # Get the SNPs with value 1
        mask_zero = sorted_gene_indices[(self.victim[sorted_gene_indices] == 0)][:1000:2]  # Get the SNPs with value 0
        
        # Use torch.cat to concatenate the indices
        queries = torch.cat((mask_one, mask_zero))
        # print(queries.size())
        # print(f"\tAttacker Queries: {queries}")
        return queries
    



