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


        self.beacon_actions =  torch.tensor([], dtype=torch.float32)
        self.attacker_actions = torch.tensor([], dtype=torch.long)
        #########################

        if self.args.attacker_type == "optimal":
            self.optimal_queries = self._init_optimal_attacker()
        
        if self.args.attacker_type == "SF":
            self.diff_discriminative_queries = self._init_diff_discriminative_attacker()

        # print("Optimal Attacker", self.optimal_queries)
        # print("diff_discriminative_queries Attacker", self.diff_discriminative_queries)

        if self.args.attacker_type == "random":
            sorted_gene_indices = torch.argsort(self.mafs)
            K = int(self.args.gene_size * 0.1)
            self.risky_queries = sorted_gene_indices[:K].tolist()
            self.non_risky_queries = sorted_gene_indices[K:].tolist()

        # if self.args.attacker_type == "agent":
        self.agent_queries = self._init_agent_attacker()

        ######################## Initializing the control LRTS
        self.control_lrts = torch.zeros(size=(self.args.a_control_size,))
        self.victim_lrt = torch.as_tensor(0)

    
    def __get_categorical(self, maf_values):
        maf_categories = torch.zeros((6), device=self.beacon_actions.device)
        
        for maf_values in maf_values:
            if maf_values < .03:
                maf_categories[0] += 1
            elif maf_values < 0.1:
                maf_categories[1] += 1
            elif maf_values < 0.2:
                maf_categories[2] += 1
            elif maf_values < 0.3:
                maf_categories[3] += 1
            elif maf_values < 0.4:
                maf_categories[4] += 1
            else:
                maf_categories[5] += 1
                
        return maf_categories

    #################################################################################################
    # Get Attacker state
    def get_state(self) -> float:
        min_control_lrt = torch.min(self.control_lrts)
        mean_control_lrt = torch.mean(self.control_lrts)
        # multiplied_values = self.victim[self.agent_queries] * self.mafs[self.agent_queries]
        victim_mafs = self.victim * self.mafs
        maf_categories_count = self.__get_categorical(victim_mafs)
        
        final_victim = torch.cat((
            self.victim_lrt.unsqueeze(0),
            min_control_lrt.unsqueeze(0),
            mean_control_lrt.unsqueeze(0),
            maf_categories_count
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
        self.victim[self.attacker_actions] = 0


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

        # TODO: Change and check this. The attacker should be able to query ever region.
        if self.args.attacker_type == "agent":
            # return self.agent_queries[agent_action]
            return agent_action # it can query everywhere
        
        # if self.args.attacker_type == "random":
        #     return torch.randint(0, self.args.gene_size, (1,)).item()

        if self.args.attacker_type == "random":
            if random.random() < self.risk_level and self.risky_queries:
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

