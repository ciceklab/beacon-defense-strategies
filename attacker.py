import numpy as np
import copy
import os
import csv
import math
import random
import bisect

import torch

from utils import calculate_ind_lrt, lrt

class Attacker():
    def __init__(self, args, victim, control, mafs):
        self.args = args
        self.mafs = mafs
        self.victim = victim
        self.attacker_control = control
        # print("Initializing {} Attacker".format(self.args.attacker_type))


        self.beacon_actions =  torch.tensor([], dtype=torch.float32)
        self.attacker_actions = torch.tensor([], dtype=torch.long)
        #########################

        if self.args.attacker_type == "optimal":
            self.optimal_queries = self._init_optimal_attacker()
        
        if self.args.attacker_type == "SF":
            self.diff_discriminative_queries = self._init_diff_discriminative_attacker()


        if self.args.attacker_type == "agent":
            self.maf_categories, self.maf_indices = self._init_agent_attacker()

        # print("Optimal Attacker", self.optimal_queries)
        # print("diff_discriminative_queries Attacker", self.diff_discriminative_queries)

        if self.args.attacker_type == "random":
            sorted_gene_indices = torch.argsort(self.mafs)
            K = int(self.args.gene_size * 0.1)
            self.risky_queries = sorted_gene_indices[:K].tolist()
            self.non_risky_queries = sorted_gene_indices[K:].tolist()

        ######################## Initializing the control LRTS
        self.control_lrts = torch.zeros(size=(self.args.a_control_size,))
        self.victim_lrt = torch.as_tensor(0)


    #################################################################################################
    # Get Attacker state
    def get_state(self) -> float:
        min_control_lrt = torch.min(self.control_lrts)
        mean_control_lrt = torch.mean(self.control_lrts)

        final_victim = torch.cat((
            self.maf_categories.flatten(),
            self.victim_lrt.unsqueeze(0),
            min_control_lrt.unsqueeze(0),
            mean_control_lrt.unsqueeze(0),
            self._calc_pvalue().unsqueeze(0)
        ), dim=0)

        return final_victim

    #################################################################################################
    # Update Attacker
    def update(self, beacon_action, attacker_action):
        beacon_action_tensor = torch.as_tensor(beacon_action, dtype=torch.float32)
        attacker_action_tensor = torch.as_tensor([attacker_action], dtype=torch.long)

        self.attacker_action = attacker_action_tensor
        self.beacon_actions = torch.cat((self.beacon_actions, beacon_action_tensor.unsqueeze(0)))
        self.attacker_actions = torch.cat((self.attacker_actions, attacker_action_tensor))

        maf = self.mafs[self.attacker_actions]
        self.victim_lrt =  self._calc_group_lrts(self.victim[self.attacker_actions], maf, self.beacon_actions, self.victim_lrt, True)
        self.control_lrts = self._calc_group_lrts(self.attacker_control[:, self.attacker_actions], maf, self.beacon_actions, self.control_lrts )
        
        # Removing the previously used SNP from the victim
        # self.victim[attacker_action] = 0

        if self.args.attacker_type == "agent":
            self.maf_categories[self._get_maf_category(
                self.mafs[attacker_action]), 0] -= 1


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

            if agent_action < 0 or agent_action >= len(self.maf_indices):
                raise ValueError(f"Invalid group number: {agent_action}. Must be between 0 and {len(self.maf_indices) - 1}.")

            if len(self.maf_indices[agent_action]) == 0:
                print(f"No indices left in group {agent_action} to sample.")
                agent_action = -1
                while len(self.maf_indices[agent_action]) == 0:
                    print(f"No indices left in group {agent_action} to sample.")
                    agent_action -= 1

                # raise ValueError(f"No indices left in group {agent_action} to sample.")
            
            sampled_index = random.choice(self.maf_indices[agent_action])
            self.maf_indices[agent_action].remove(sampled_index)
            return sampled_index
    

        if self.args.attacker_type == "random":
            if random.random() < 0.3 and self.risky_queries:
                query = random.choice(self.risky_queries)
                self.risky_queries.remove(query)
            elif self.non_risky_queries:
                query = random.choice(self.non_risky_queries)
                self.non_risky_queries.remove(query)
            else:
                query = None
            return query

        if self.args.attacker_type == "SF":
            return self.diff_discriminative_queries[current_step]
        


    #################################################################################################
    def _calc_group_lrts_all_snps(self, genome, maf, response) -> torch.Tensor:
        error = 0.001

        one_minus_maf = (1 - maf)
        DN_i = one_minus_maf.pow(2 * self.args.beacon_size)
        DN_i_1 = one_minus_maf.pow(2 * self.args.beacon_size - 2)

        log1 = torch.log(DN_i) - torch.log(error * DN_i_1)
        log2 = torch.log((error * DN_i_1) * (1 - DN_i)) - torch.log(DN_i * (1 - error * DN_i_1))

        lrt = (log1 + log2 * response).unsqueeze(0) * genome 
        return lrt
    #LRT PVALUES
    def _calc_group_lrts(self, genome, maf, response, prev_beacon_lrts, one_dim=False) -> torch.Tensor:
        error = 0.001

        one_minus_maf = (1 - maf[-1])  # Last MAF value
        DN_i = one_minus_maf.pow(2 * self.args.beacon_size)
        DN_i_1 = one_minus_maf.pow(2 * self.args.beacon_size - 2)

        log1 = torch.log(DN_i) - torch.log(error * DN_i_1)
        log2 = torch.log((error * DN_i_1) * (1 - DN_i)) - torch.log(DN_i * (1 - error * DN_i_1))

        if one_dim:
            last_genome=genome[-1]
        else:
            last_genome=genome[:, -1]

        lrt = (log1 + log2 * response[-1]).mul(last_genome)
        updated_beacon_lrts = prev_beacon_lrts + lrt

        return updated_beacon_lrts
    
    
    def _calc_pvalue(self):
        victim_lrt = copy.deepcopy(self.victim_lrt)
        control_lrts = copy.deepcopy(self.control_lrts)

        pvalue=torch.sum(victim_lrt >= control_lrts) / self.args.a_control_size
        return torch.Tensor(pvalue)


    #################################################################################################
    # Optimal Attacker
    def _init_optimal_attacker(self):
        maf_values = self.mafs

        maf_mask = torch.logical_or(maf_values >= 0.05, maf_values == 0.001)
        filtered_indices = torch.argsort(maf_values[maf_mask])
        sorted_gene_indices = torch.nonzero(maf_mask).squeeze()[filtered_indices]
    
        # print("\tInitializing the Optimal Attack Strategy")
        # sorted_gene_indices = torch.argsort(maf_values)  # Sort the MAFs using PyTorch
        # print("Victim : {}".format(self.victim))
        # print("Victim Sorted Genes: {}".format(sorted_gene_indices))
        
        mask_one = (self.victim[sorted_gene_indices] == 1)  # Get the SNPs with value 1
        mask_zero = (self.victim[sorted_gene_indices] == 0)  # Get the SNPs with value 0
        
        # Use torch.cat to concatenate the indices
        queries = torch.cat((sorted_gene_indices[mask_one], sorted_gene_indices[mask_zero]))#[:self.args.max_queries]
        
        # print(f"\tAttacker Queries: {queries}")
        return queries
    

    # def _init_agent_attacker(self):
    #     # print("\tInitializing the Optimal Attack Strategy")
    #     maf_values = self.mafs
    #     sorted_gene_indices = torch.argsort(maf_values)  # Sort the MAFs using PyTorch
    #     # print("Victim : {}".format(self.victim))
    #     # print("Victim Sorted Genes: {}".format(sorted_gene_indices))
        
    #     mask_one = sorted_gene_indices[(self.victim[sorted_gene_indices] == 1)][:1000:2] # Get the SNPs with value 1
    #     mask_zero = sorted_gene_indices[(self.victim[sorted_gene_indices] == 0)][:1000:2]  # Get the SNPs with value 0
        
    #     # Use torch.cat to concatenate the indices
    #     queries = torch.cat((mask_one, mask_zero))
    #     # print(queries.size())
    #     # print(f"\tAttacker Queries: {queries}")
    #     return queries

    def _init_diff_discriminative_attacker(self, k=5):
        victim_lrts = self._calc_group_lrts_all_snps(self.victim, self.mafs, 1)
        control_lrts = self._calc_group_lrts_all_snps(self.attacker_control, self.mafs, 1)

        discriminative_powers = victim_lrts.mean(dim=0) - control_lrts.mean(dim=0)

        flipped_victim_lrts = self._calc_group_lrts_all_snps(self.victim, self.mafs, 0)
        flipped_control_lrts = self._calc_group_lrts_all_snps(self.attacker_control, self.mafs, 0)
        flipped_discriminative_powers = flipped_victim_lrts.mean(dim=0) - flipped_control_lrts.mean(dim=0)

        delta_discriminative_powers = discriminative_powers - flipped_discriminative_powers

        sorted_gene_indices = torch.argsort(delta_discriminative_powers, descending=False)
        mask_one = (self.victim[sorted_gene_indices] == 1)  # Get the SNPs with value 1
        mask_zero = (self.victim[sorted_gene_indices] == 0)  # Get the SNPs with value 0
        queries = torch.cat((sorted_gene_indices[mask_one], sorted_gene_indices[mask_zero]))#[:self.args.max_queries]
        print("Attacker Queries: ", sorted_gene_indices)
        return queries


    def _get_maf_category(self, maf_value):
        for i in range(len(self.maf_thresholds) - 1):
            if self.maf_thresholds[i] <= maf_value < self.maf_thresholds[i + 1]:
                return i
        return len(self.maf_thresholds) - 1 


    def _init_agent_attacker(self):
        num_groups = 99 
        
        maf_values_victim = self.mafs * self.victim
        maf_values_victim = maf_values_victim[maf_values_victim != 0]
        
        self.maf_thresholds, group_counts = self._divide_mafs(maf_values_victim, num_groups)
        print(f"Divided MAFs into {num_groups} groups with: \n\tThresholds: {self.maf_thresholds} \n\tGroup Counts: {group_counts}")


        maf_categories = torch.zeros((num_groups + 1, 5), device=self.beacon_actions.device)
        maf_indices = [[] for _ in range(num_groups + 1)]
        victim_mask = self.victim == 1

        for idx, maf_value in enumerate(self.mafs):
            if victim_mask[idx]:  
                category = self._get_maf_category(maf_value)
                maf_categories[category, 0] += 1
                maf_indices[category].append(idx)
            else:
                maf_categories[-1, 0] += 1
                maf_indices[-1].append(idx)

        for i, maf_list in enumerate(maf_indices):
            if len(maf_list) > 0:
                maf_tensor = torch.tensor(self.mafs[maf_list])
                min_maf, max_maf = torch.min(maf_tensor), torch.max(maf_tensor)
                # print(min_maf, max_maf)

                lrt_min_1 = lrt(self.args.a_control_size, 1, min_maf, torch.tensor(1))
                lrt_max_1 = lrt(self.args.a_control_size, 1, max_maf, torch.tensor(1))

                lrt_min_0 = lrt(self.args.a_control_size, 1, min_maf, torch.tensor(0))
                lrt_max_0 = lrt(self.args.a_control_size, 1, max_maf, torch.tensor(0))

                maf_categories[i, 1] = lrt_min_1
                maf_categories[i, 2] = lrt_max_1
                maf_categories[i, 3] = lrt_min_0
                maf_categories[i, 4] = lrt_max_0
        # print(maf_categories)
        return maf_categories, maf_indices
    
    def _divide_mafs(self, numbers, num_groups):
        sorted_numbers = torch.sort(numbers).values

        n = len(sorted_numbers)
        split_indices = [int(n * i / num_groups) for i in range(1, num_groups)]
        group_boundaries = [sorted_numbers[0].item()] + [sorted_numbers[idx].item() for idx in split_indices] + [sorted_numbers[-1].item()]
        group_boundaries = [round(bound, 3) if bound > 1e-6 else 0.0 for bound in group_boundaries]
        group_counts = []
        
        for i in range(num_groups):
            if i == 0:
                count = torch.sum(numbers < group_boundaries[i+1]).item()
            elif i == num_groups - 1:
                count = torch.sum(numbers >= group_boundaries[i]).item()
            else:
                count = torch.sum((numbers >= group_boundaries[i]) & (numbers < group_boundaries[i+1])).item()
            
            group_counts.append(count)
        
        return group_boundaries, group_counts

