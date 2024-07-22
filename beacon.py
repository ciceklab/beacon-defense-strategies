import numpy as np
import copy
import os
import csv
import math
import random

import torch

from utils import calculate_ind_lrt

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
        self.sum_probs = 0
        self.lie_probs = 0

        ######################## Initializing the control and case LRTS
        self.beacon_lrts = torch.zeros(size=(self.args.beacon_size,))
        self.control_lrts = torch.zeros(size=(self.args.b_control_size,))
        self.pvalues = self._calc_pvalues()

        
    #################################################################################################
    # Get Beacon state
    def get_state(self, attacker_action)->float:
        # print("Beacon Side: ")

        min_pvalue = torch.min(self.pvalues)
        min_case = torch.min(self.beacon_lrts)
        max_case = torch.max(self.beacon_lrts)
        thresh_lrt = torch.sort(self.control_lrts)[0][int(0.05*self.args.b_control_size)]
        min_control = torch.min(self.control_lrts)
        max_control = torch.max(self.control_lrts)
        victim_pvalue = self.pvalues[self.victim_id]


        # print("\tControls min: {} and max {} LRT".format(min_control, max_control,))
        # print("\tBeacons min: {} and max: {} LRT".format( min_case, max_case))
        # print("\tVictim's LRT: {}".format(self.beacon_lrts[self.victim_id]))
        # print("\tThreshold LRT: {}".format(thresh_lrt))
        # print("\tVictim's Pvalue: {}".format(victim_pvalue))

        return [self.mafs[attacker_action], min_pvalue, min_case, max_case, thresh_lrt, min_control, max_control, torch.min(self.mafs), torch.max(self.mafs)]


    #################################################################################################
    # Update Beacon
    def update(self, beacon_action, attacker_action):
        # Update the beacon case group and calculate the LRT values
        self.beacon_info[:, attacker_action, 2] = torch.as_tensor(beacon_action)
        self.beacon_lrts = self._calc_group_lrts(self.beacon_info, True)

        # Update the beacon control group and calculate the LRT values
        self.control_info[:, attacker_action, 2] = torch.as_tensor(1)
        self.control_lrts = self._calc_group_lrts(self.control_info, True)

        self.sum_probs += beacon_action
        self.lie_probs += beacon_action if beacon_action < 0.5 else 0

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


    #################################################################################################
    # Beacon Action
    def act(self, attacker_action, max_pvalue_change_threshold):
        if self.args.beacon_type == "random":
            return random.random()
        if self.args.beacon_type == "truth":
            return 1
        if self.args.beacon_type == "beacon_strategy":
            return self.beacon_strategy_pvalue_change(self.beacon_case, attacker_action, max_pvalue_change_threshold)

    # Beacon Strategy: Flips the answer if any of the plvalues is smaller than the threshold
    def beacon_strategy_pvalue(self, beacon_case, snp_position, threshold = 0.05):

        beacon_action = int(torch.any(beacon_case[:, snp_position] == 1).item())
        self.update(beacon_action, snp_position)
        pvalues = self._calc_pvalues()

        if beacon_action == 1:
            if torch.min(pvalues) < threshold:
                beacon_action = 0
                self.update(beacon_action, snp_position)

        return beacon_action



    #Beacon Strategy: Flips the answer if any of the plvalue changes are larger than the threshold
    def beacon_strategy_pvalue_change(self, beacon_case, snp_position, max_pvalue_change_threshold = 0.3):

        beacon_action = int(torch.any(beacon_case[:, snp_position] == 1).item())
        print(beacon_action)
        if beacon_action == 1:
            initial_pvalues = self._calc_pvalues()
            self.update(beacon_action, snp_position)
            new_pvalues= self._calc_pvalues()

            max_pvalue_change = torch.max(torch.abs(new_pvalues - initial_pvalues)).item()

            if max_pvalue_change > max_pvalue_change_threshold:
                beacon_action = 0
                self.update(beacon_action, snp_position)


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

        

