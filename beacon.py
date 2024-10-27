import numpy as np
import copy
import os
import csv
import math
import random

import torch

from utils import calculate_ind_lrt, calculate_pvalues

class Beacon():
    def __init__(self, args, case, control, mafs, victim_id):
        self.args = args
        self.mafs = mafs
        self.beacon_case = case
        self.beacon_control = control
        self.victim_id = victim_id
        # print("Initializing {} Beaocn".format(self.args.beacon_type))

        self.attacker_actions = torch.tensor([], dtype=torch.long)
        self.responses = torch.tensor([], dtype=torch.float32)

        ######################## For calculating Utility Reward 
        self.sum_probs = 1

        ######################## Initializing the control and case LRTS
        self.beacon_lrts = torch.zeros(size=(self.args.beacon_size,))
        self.control_lrts = torch.zeros(size=(self.args.b_control_size,))
        self.pvalues = self._calc_pvalues()


        ######################## Init Beacons
        if self.args.beacon_type == "baseline":
            self.baseline_mafs = self._init_baseline_beaon()

        if self.args.beacon_type == "strategic":
            self.strategy_positions = self._init_strategic_beaon()

        if self.args.beacon_type == "qbudget":
            p = 0.1
            initial_budget = -torch.log(torch.tensor(p))
            self.budgets = torch.full(size=(self.args.beacon_size,), fill_value=initial_budget)
        


        
    #################################################################################################
    # Get Beacon state
    def get_state(self, attacker_action, current_step)->float:
        # print("Beacon Side: ")

        b_lrts_afterN, c_lrts_afterN=self.update(1, attacker_action, calculation_mode=True)
        pvalues_afterN = calculate_pvalues(b_lrts_afterN, c_lrts_afterN, self.args.b_control_size)

        b_lrts_after, c_lrts_after=self.update(0, attacker_action, calculation_mode=True)
        pvalues_after = calculate_pvalues(b_lrts_after, c_lrts_after, self.args.b_control_size)


        # Case group information
        min_lrt_case = torch.min(self.beacon_lrts)
        mean_lrt_case = torch.mean(self.beacon_lrts)
        min_lrt_case_afterN = torch.min(b_lrts_afterN)
        mean_lrt_case_afterN = torch.mean(b_lrts_afterN)
        min_lrt_case_after = torch.min(b_lrts_after)
        mean_lrt_case_after = torch.mean(b_lrts_after)
        min_pvalue = torch.min(self.pvalues)
        min_pvalue_after = torch.min(pvalues_after)
        min_pvalue_afterN = torch.min(pvalues_afterN)

        # Control group information
        min_lrt_control = torch.min(self.control_lrts)
        mean_lrt_control = torch.mean(self.control_lrts)
        thresh_lrt = torch.sort(self.control_lrts)[0][int(0.05*self.args.b_control_size)]
        min_lrt_control_afterN = torch.min(c_lrts_afterN)
        mean_lrt_control_afterN = torch.mean(c_lrts_afterN)
        min_lrt_control_after = torch.min(c_lrts_after)
        mean_lrt_control_after = torch.mean(c_lrts_after)


        return [self.mafs[attacker_action], min_lrt_case, mean_lrt_case, min_lrt_case_afterN, mean_lrt_case_afterN, min_lrt_case_after, mean_lrt_case_after, min_pvalue, min_pvalue_after, min_pvalue_afterN, min_lrt_control, mean_lrt_control, thresh_lrt, min_lrt_control_afterN, mean_lrt_control_afterN, min_lrt_control_after, mean_lrt_control_after, self.sum_probs/(current_step+1)]


    #################################################################################################
    # Update Beacon
    def update(self, beacon_action, attacker_action, calculation_mode: bool = False):
        beacon_action_tensor = torch.as_tensor(beacon_action, dtype=torch.float32)
        attacker_action_tensor = torch.as_tensor([attacker_action], dtype=torch.long)

        if calculation_mode:
            attacker_actions = torch.cat((self.attacker_actions, attacker_action_tensor))
            res = torch.cat((self.responses, beacon_action_tensor.unsqueeze(0)))

            b_lrts = self._calc_group_lrts(self.beacon_case[:, attacker_actions], self.mafs[attacker_actions], res, self.beacon_lrts.clone())
            c_lrts = self._calc_group_lrts(self.beacon_control[:, attacker_actions], self.mafs[attacker_actions], res, self.control_lrts.clone())

            return b_lrts, c_lrts


        self.attacker_actions = torch.cat((self.attacker_actions, attacker_action_tensor))
        self.responses = torch.cat((self.responses, beacon_action_tensor.unsqueeze(0)))

        maf = self.mafs[self.attacker_actions]
        self.beacon_lrts = self._calc_group_lrts(self.beacon_case[:, self.attacker_actions], maf, self.responses, self.beacon_lrts, update_qb=True)
        self.control_lrts = self._calc_group_lrts(self.beacon_control[:, self.attacker_actions], maf, self.responses, self.control_lrts)

        if isinstance(beacon_action, list):
            self.sum_probs += beacon_action[0]
        else:
            self.sum_probs += beacon_action

        # print(self.beacon_case[:, self.attacker_actions])
        # print("self.beacon_lrts ", self.beacon_lrts)
        # print(self.beacon_control[:, self.attacker_actions])
        # print("self.control_lrts ", self.control_lrts)

        

    #################################################################################################
    # Reward Calculation
    def calc_reward(self, beacon_action)->float:
        done = False
        self.pvalues = self._calc_pvalues()
        # preward = torch.min(self.pvalues)
        preward = self.pvalues[self.victim_id]
        # ureward = torch.Tensor([0]) if self.sum_probs == 0 else 1-(self.lie_probs / self.sum_probs)
        ureward = beacon_action
        total = ureward + preward
    
        if preward < 0.05:
            total -= 15
            done=True
            # print("⛔⛔⛔ Attacker Identified the VICTIM ⛔⛔⛔")

        # print("Beacon Rewards: \n\tprivacy: {}\n\tutility: {}\n\ttotal: {}".format(preward, ureward, total))
        # print("-----------------------------------------------------------------")
        return total, done, [preward, ureward]


    #################################################################################################
    # Beacon Action
    def act(self, attacker_action, max_pvalue_change_threshold):
        if self.args.beacon_type == "random":
            has_snp = torch.sum(self.beacon_case[:, attacker_action]) > 0
            if not has_snp:
                return 1
            else:
                if random.random() < 0.75:
                    return 1
                else:
                    return 0
        if self.args.beacon_type == "truth":
            return 1
        if self.args.beacon_type == "beacon_strategy":
            return self.beacon_strategy_pvalue_change(self.beacon_case, attacker_action, max_pvalue_change_threshold)
        if self.args.beacon_type == "baseline":
            if self.mafs[attacker_action] in self.baseline_mafs: 
                return 0
            else: 
                return 1
        if self.args.beacon_type == "qbudget":
            # print("Before", self.budgets)
            self._update_budget(attacker_action, self.mafs[attacker_action])
            # print("After", self.budgets)
            has_snp = torch.sum(self.beacon_case[:, attacker_action]) > 0
            if not has_snp:
                # print("No SNP in Beacon")
                return 1
            valid_individuals = self.budgets > 0
            remaining_genomes = self.beacon_case[valid_individuals, attacker_action]
            has_snp_after_budget_check = torch.sum(remaining_genomes) > 0
            if not has_snp_after_budget_check:
                # print("No SNP in Beacon After Budget")
                return 0
            # print("In the Beacon")
            return 1
        
        if self.args.beacon_type == "strategic":
            if attacker_action in self.strategy_positions: 
                return 0
            else: 
                return 1

    def _init_strategic_beaon(self, k=0.05):
        beacon_lrts = self._calc_group_lrts_all_snps(self.beacon_case, self.mafs, 1)
        control_lrts = self._calc_group_lrts_all_snps(self.beacon_control, self.mafs, 1)
        discriminative_powers = beacon_lrts.mean(dim=0) - control_lrts.mean(dim=0)
        flipped_beacon_lrts = self._calc_group_lrts_all_snps(self.beacon_case, self.mafs, 0)
        flipped_control_lrts = self._calc_group_lrts_all_snps(self.beacon_control, self.mafs, 0)
        flipped_discriminative_powers = flipped_beacon_lrts.mean(dim=0) - flipped_control_lrts.mean(dim=0)

        delta_discriminative_powers = discriminative_powers - flipped_discriminative_powers
        sorted_gene_indices = torch.argsort(delta_discriminative_powers, descending=False)

        # num_snps_to_flip = int(len(delta_discriminative_powers) * (k / 100))
        # top_k_indices = torch.topk(delta_discriminative_powers, num_snps_to_flip).indices
        # print("top_k_indices: ", top_k_indices)
        K = int(self.args.gene_size * k)
        print("Beacon Queries: ", sorted_gene_indices[:K])

        return sorted_gene_indices[:K]


    def _init_baseline_beaon(self, k=10):
        un_mafs = torch.unique(torch.as_tensor(self.mafs))
        return un_mafs[1:int(k / 100 * un_mafs.numel())]

    #################################################################################################
    #LRT PVALUES
    def _calc_group_lrts_all_snps(self, genome, maf, response) -> torch.Tensor:
        error = 0.001

        one_minus_maf = (1 - maf)
        DN_i = one_minus_maf.pow(2 * self.args.beacon_size)
        DN_i_1 = one_minus_maf.pow(2 * self.args.beacon_size - 2)

        log1 = torch.log(DN_i) - torch.log(error * DN_i_1)
        log2 = torch.log((error * DN_i_1) * (1 - DN_i)) - torch.log(DN_i * (1 - error * DN_i_1))

        lrt = (log1 + log2 * response).unsqueeze(0) * genome 
        return lrt

    def _calc_group_lrts(self, genome, maf, response, prev_beacon_lrts, update_qb=False) -> torch.Tensor:
        error = 0.001

        one_minus_maf = (1 - maf[-1])  # Last MAF value
        DN_i = one_minus_maf.pow(2 * self.args.beacon_size)
        DN_i_1 = one_minus_maf.pow(2 * self.args.beacon_size - 2)

        log1 = torch.log(DN_i) - torch.log(error * DN_i_1)
        log2 = torch.log((error * DN_i_1) * (1 - DN_i)) - torch.log(DN_i * (1 - error * DN_i_1))

        lrt = (log1 + log2 * response[-1]).mul(genome[:, -1])
        updated_beacon_lrts = prev_beacon_lrts + lrt

        return updated_beacon_lrts



    def _update_budget(self, attacker_action, maf):
        # Get a mask of individuals who have the SNP at the attacker_action position
        people_with_snp = self.beacon_case[:, attacker_action] > 0
        valid_individuals = people_with_snp & (self.budgets > 0)
        
        # If there are valid individuals, update their budgets
        if torch.sum(valid_individuals) > 0:
            budget_costs = self._calculate_budget_cost(maf)
            # Update budgets for individuals who have the SNP and valid budgets
            self.budgets[valid_individuals] -= budget_costs


    def _calc_pvalues(self):
        beacon_lrts = copy.deepcopy(self.beacon_lrts)
        control_lrts = copy.deepcopy(self.control_lrts)

        pvalues=[]
        for blrt in beacon_lrts:
            pvalue=torch.sum(blrt >= control_lrts) / self.args.b_control_size
            pvalues.append(pvalue)
        
        if torch.any(torch.Tensor(pvalues) < 0):
            print(beacon_lrts, control_lrts)
            assert "Wrong p value"

        return torch.Tensor(pvalues)

    def _calc_percentage_of_snps_at_position(self, people, position):
        if position < 0 or position >= people.size(1):
            raise ValueError("Index n is out of bounds for the number of positions.")

        count_ones = torch.sum(people[:, position])
        total_people = people.size(0)         

        percentage = count_ones / total_people
        # print(count_ones, total_people, percentage)
        return percentage 


    def _calculate_budget_cost(self, maf_value):
        D_qi = (1 - maf_value).pow(self.args.beacon_size - 1)
        return -torch.log(1 - D_qi)
