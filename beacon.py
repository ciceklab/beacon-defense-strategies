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


        ######################## Init the Beacon info
        temp_maf = torch.Tensor(self.mafs).unsqueeze(0).expand(self.args.beacon_size, -1)
        responses = torch.ones(size=(self.args.beacon_size, self.args.gene_size))*-1
        current_query = torch.zeros(size=(self.args.beacon_size, self.args.gene_size))
        lrts = torch.zeros(size=(self.args.beacon_size, self.args.gene_size))
        self.beacon_info = torch.stack([torch.Tensor(self.beacon_case), temp_maf, responses, current_query, lrts], dim=-1)

        ######################## Init the Control info
        temp_maf = torch.Tensor(self.mafs).unsqueeze(0).expand(self.args.b_control_size, -1)
        responses = torch.ones(size=(self.args.b_control_size, self.args.gene_size))*-1
        current_query = torch.ones(size=(self.args.b_control_size, self.args.gene_size))
        lrts = torch.zeros(size=(self.args.b_control_size, self.args.gene_size))
        self.control_info = torch.stack([torch.Tensor(self.beacon_control), temp_maf, responses, current_query, lrts], dim=-1)

        ######################## For calculating Utility Reward 
        self.sum_probs = 1

        ######################## Initializing the control and case LRTS
        self.beacon_lrts = torch.zeros(size=(self.args.beacon_size,))
        self.control_lrts = torch.zeros(size=(self.args.b_control_size,))
        self.pvalues = self._calc_pvalues()

        
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
        



        # print("\tControls min: {} and max {} LRT".format(min_control, max_control,))
        # print("\tBeacons min: {} and max: {} LRT".format( min_case, max_case))
        # print("\tVictim's LRT: {}".format(self.beacon_lrts[self.victim_id]))
        # print("\tThreshold LRT: {}".format(thresh_lrt))
        # print("\tVictim's Pvalue: {}".format(victim_pvalue))
        # print("\tCase percentage: {}".format(case_per))
        # print("\Control percentage: {}".format(control_case))


        return [self.mafs[attacker_action], min_lrt_case, mean_lrt_case, min_lrt_case_afterN, mean_lrt_case_afterN, min_lrt_case_after, mean_lrt_case_after, min_pvalue, min_pvalue_after, min_pvalue_afterN, min_lrt_control, mean_lrt_control, thresh_lrt, min_lrt_control_afterN, mean_lrt_control_afterN, min_lrt_control_after, mean_lrt_control_after, self.sum_probs/(current_step+1)]


    #################################################################################################
    # Update Beacon
    def update(self, beacon_action, attacker_action, calculation_mode: bool = False):
        if calculation_mode:
            b_info = copy.deepcopy(self.beacon_info)
            b_info[:, attacker_action, 2] = torch.as_tensor(beacon_action)
            b_lrts = self._calc_group_lrts(b_info, False)
            
            c_info = copy.deepcopy(self.control_info)
            c_info[:, attacker_action, 2] = torch.as_tensor(1)
            c_lrts = self._calc_group_lrts(c_info, False)

            return b_lrts, c_lrts


        # Update the beacon case group and calculate the LRT values
        self.beacon_info[:, attacker_action, 2] = torch.as_tensor(beacon_action)
        self.beacon_lrts = self._calc_group_lrts(self.beacon_info, True)

        # Update the beacon control group and calculate the LRT values
        self.control_info[:, attacker_action, 2] = torch.as_tensor(1)
        self.control_lrts = self._calc_group_lrts(self.control_info, True)
        # print(beacon_action)
        if isinstance(beacon_action, list):
            self.sum_probs += beacon_action[0]
        else:
            self.sum_probs += beacon_action
        

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
            print("⛔⛔⛔ Attacker Identified the VICTIM ⛔⛔⛔")

        # print("Beacon Rewards: \n\tprivacy: {}\n\tutility: {}\n\ttotal: {}".format(preward, ureward, total))
        # print("-----------------------------------------------------------------")
        return total, done, [preward, ureward]

    # def calc_reward(self, beacon_action):
    #     done = False
    #     responses = self.beacon_info[:, :, 2]
    #     #num_responses = (responses != -1).sum().item()
    #     si = responses #self.sum_probs / num_responses
    #     lrt_values = self.beacon_lrts
    #     lrt_values = lrt_values.numpy()

    #     # print(lrt_values)

    #     q1 = np.quantile(lrt_values, 0.25)
    #     target_lrts = lrt_values[lrt_values < q1]
    #     if len(target_lrts) != 0:
    #         target_lrts = np.append(target_lrts, np.mean(lrt_values))
    #         target_lrts = (target_lrts - target_lrts.min()) / \
    #                 (target_lrts.max() - target_lrts.min())

    #     mean = np.mean(target_lrts)
    #     std_dev = np.std(target_lrts, ddof=1)

    #     coe_variance = (std_dev/mean)
    #     if std_dev < 1 and mean < 1:
    #         coe_variance = std_dev
    #     term_one = (1 - coe_variance
    #            ) if mean != 0 and len(target_lrts) != 0 else 1

    #     mask = (si != -1)
    #     non_neg_si = si[mask]
    #     term_two = torch.sum(non_neg_si) / len(si)
    #     total = (term_one * 5 + term_two) / 6
    #     #print("term_one", term_one, "term_two", term_two)

    #     if self.pvalues[self.victim_id] < 0.05:
    #         #total -= 15
    #         done = True
    #         print("⛔⛔⛔ Attacker Identified the VICTIM ⛔⛔⛔")

    #     # print("beacon_reward", total)
    #     return total, done, [term_one, term_two]


    #################################################################################################
    # Beacon Action
    def act(self, attacker_action, max_pvalue_change_threshold):
        if self.args.beacon_type == "random":
            return random.random()
        if self.args.beacon_type == "urandom":
            random_number = np.random.normal(0.8, 0.1)
            random_number = np.clip(random_number, 0, 1)
            return random_number
        if self.args.beacon_type == "truth":
            return 1
        if self.args.beacon_type == "beacon_strategy":
            return self.beacon_strategy_pvalue_change(self.beacon_case, attacker_action, max_pvalue_change_threshold)

    # Beacon Strategy: Flips the answer if any of the plvalues is smaller than the threshold
    def beacon_strategy_pvalue(self, beacon_case, snp_position, threshold = 0.05):

        beacon_action = int(torch.any(beacon_case[:, snp_position] == 1).item())
        self.update(beacon_action, snp_position, calculation_mode=True)
        pvalues = self._calc_pvalues()

        if beacon_action == 1:
            if torch.min(pvalues) < threshold:
                beacon_action = 0
        return beacon_action


    #Beacon Strategy: Flips the answer if any of the plvalue changes are larger than the threshold
    def beacon_strategy_pvalue_change(self, beacon_case, snp_position, max_pvalue_change_threshold = 0.3):
        beacon_action = int(torch.any(beacon_case[:, snp_position] == 1).item())

        if beacon_action == 1:
            initial_pvalues = self._calc_pvalues()
            self.update(beacon_action, snp_position, calculation_mode=True)
            new_pvalues= self._calc_pvalues()
            max_pvalue_change = torch.max(torch.abs(new_pvalues - initial_pvalues)).item()

            if max_pvalue_change > max_pvalue_change_threshold:
                beacon_action = 0

        return beacon_action

    #################################################################################################
    #LRT PVALUES
    def _calc_group_lrts(self, group, save=True)->float:
        lrt_values = []
        for index, individual in enumerate(group):
            lrts = calculate_ind_lrt(group[index, :, :], gene_size=self.args.gene_size, number_of_people=self.args.beacon_size)

            lrt_values.append(torch.sum(lrts)) ############ Sum of the LRTs
            if save:
                group[index, :, 4] = torch.Tensor(lrts)

        return torch.Tensor(lrt_values)
    
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
