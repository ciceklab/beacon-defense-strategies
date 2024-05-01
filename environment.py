import numpy as np
import copy
import os
import csv
import math
import random

from gym import Env
from gym import spaces

import torch

class BeaconEnv(Env):
    def __init__(self, args, beacon, maf, binary):
        super(BeaconEnv, self).__init__()

        self.beacon=beacon
        self.maf=maf
        self.args=copy.copy(args)
        self.binary=binary


        # Randomly set populations and genes
        self.s_beacon, self.s_control, self.a_control, self.victim, self.mafs = self.get_populations()
        # print("VICTIM: {}".format(self.victim))

        self.reset_counter = -1

        # Initialize the agents
        self._init_attacker()
        self._init_beacon()
        self._init_control()



        self.attacker_action = 0
        self.beacon_action = 0
        self.altered_probs = 0
        

        # Define action and observation space
        self.action_space = spaces.Box(low=0, high=1, shape=(1,))  # Beacon lies or not
        self.observation_space = spaces.Box(low=-1, high=1,shape=(args.beacon_size, args.gene_size, 4))  # State: [Beacon Size, Gene size, 3(SNP, MAF, RES, CURRENT)]
        self.max_steps = args.max_queries  # Maximum number of steps per episode
        self.current_step = 0

        if self.args.attacker_type == "optimal":
            self.optimal_queries = self._init_optimal_attacker()
            # print(self.optimal_queries)
        self._calc_control_lrts()


        log_env_name = args.results_dir + "/env" + '/PPO_' + str(len(next(os.walk(args.results_dir))[2])) + ".csv"
        self.log_env = csv.writer(open(log_env_name,"w+"))
        self.log_env.writerow(["Episode", "Beacon", "Gene", "SNP", "MAF", "RES", "LRT", "Pvalue"])
        #log_f.write('Start\n')



    # Reset the environment after an episode
    def reset(self, resett=False) -> torch.Tensor:
        # print("RESET ENV")
        self.reset_counter+=1

        if (self.reset_counter%self.args.pop_reset_freq==0 and self.reset_counter>0) or resett :
            print("Reseting the Populations")
            self._reset_populations()
            self.reset_counter = 0

        self.altered_probs = 0
        self.current_step = 0
        self.lie_probs = 0

        # Reset the states of our agents
        self._init_attacker()
        self._init_beacon()
        self._init_control()

        if (self.reset_counter%self.args.pop_reset_freq==0 and self.reset_counter>0) or resett:
            if self.args.attacker_type == "optimal":
                self.optimal_queries = self._init_optimal_attacker()
            self._calc_control_lrts()


        if self.current_step==0:
            self._act_attacker()
            current_beacon_s = copy.deepcopy(self.beacon_state)
            current_beacon_s[:, self.attacker_action, 3] = 1
            # print(self.control_lrts[:, 0].size())

            # current_control=copy.deepcopy(self.scontrol_state)
            # self.scontrol_state[:,:,2] = (self.control_responses[:, self.current_step].view(self.args.control_size, 1).repeat(1, self.args.gene_size))
            self.scontrol_state[:, :, 4] = (self.control_lrts[:, self.current_step].view(self.args.control_size, 1).repeat(1, self.args.gene_size))

            observation = torch.concatenate([current_beacon_s, self.scontrol_state])
            # print("Observation: {}".format(observation))

            return self.attacker_state, observation

        observation = torch.concatenate([self.beacon_state, self.scontrol_state])
        # print("Observation: {}".format(observation))

        return self.attacker_state, observation

    def step(self, beacon_action:float):
        beacon_action = np.clip(beacon_action, 0, 1)
        done = False
        # # Change the res of the asked gene to 1 in the state of beacon
        # if beacon_action > 0.5:
        self.beacon_state[:, self.attacker_action, 2] = torch.Tensor(beacon_action) #TRUTH
        # self.scontrol_state[:, self.attacker_action, 2] = torch.Tensor(beacon_action) #TRUTH
        
        # else:
        #     self.beacon_state[:, self.attacker_action, 2] = -1 #LIE



        # Calculate the lrt for individuals in the beacon and find the min 
        self.altered_probs += beacon_action
        self.lie_probs += beacon_action if beacon_action < 0.5 else 0

        pvalues = self._calc_pvalues()
        min_pvalue = torch.min(pvalues)

        if torch.any(torch.isnan(pvalues)):
            print("pvalues contains NaN values#############################################")

        self.pvalues = pvalues
        # print("pvalues: ", pvalues)
        # print("Stateee: ", torch.concatenate([self.beacon_state, self.scontrol_state]))

        # lrt_values = self._calc_group_lrts(self.beacon_state)
        # self._calc_group_lrts(self.scontrol_state, False)
        # min_lrt = min(lrt_values)


        # if torch.any(torch.isnan(torch.Tensor(lrt_values))):
        #     print("lrt_values contains NaN values#############################################")


        # lrt_thresh = self._calc_lrt_threshold()
        # threshold = min_pvalue < 0.05

        preward = min_pvalue
        ureward = torch.Tensor([0]) if self.altered_probs == 0 else 1-(self.lie_probs / self.altered_probs)
    

        # print("lrt: ", self._calc_beacon_reward())
        # print("preward: ", preward)
        # print("ureward: ", ureward)
        reward = preward + ureward


        # reward += -10 * torch.sum(torch.Tensor(lrt_values) < lrt_thresh) if threshold else 0

        self.current_step += 1
        if self.current_step > self.max_steps or self.current_step == len(self.optimal_queries) or min_pvalue < 0.05:
            done = True
            observation = torch.concatenate([self.beacon_state, self.scontrol_state]) # Attacker's state will be added for the multi agent RL

        else:
            # Take attacker Action
            self._act_attacker()
            current_beacon_s = copy.deepcopy(self.beacon_state)
            current_beacon_s[:, self.attacker_action, 3] = 1
            # current_control=copy.deepcopy(self.scontrol_state)
            # self.scontrol_state[:,:,2] = (self.control_responses[:, self.current_step].view(self.args.control_size, 1).repeat(1, self.args.gene_size))
            self.scontrol_state[:, :, 4] = (self.control_lrts[:, self.current_step].view(self.args.control_size, 1).repeat(1, self.args.gene_size))


            observation = torch.concatenate([current_beacon_s, self.scontrol_state])# Attacker's state will be added for the multi agent RL
        # print("Observation: {}".format(observation))
        return observation, reward, done, [preward, ureward], pvalues
    
    def _act_attacker(self)->None:
        if self.args.attacker_type == "random":
            self.attacker_action = np.random.randint(low=0, high=self.args.gene_size)
        elif self.args.attacker_type == "optimal":
            # print("Current Step: {}".format(self.current_step))
            self.attacker_action = self.optimal_queries[self.current_step]
        else:
            raise NotImplemented
        print("Step {} Attacker Attacked {} Query".format(self.current_step, self.attacker_action))

    def _reset_populations(self)->None:
        self.s_beacon, self.s_control, self.a_control, self.victim, self.mafs = self.get_populations()
        # print("VICTIM: {}".format(self.victim))

            # Initilize the attacker states
    def _init_attacker(self)->None:
        self.attacker_state = torch.tensor([self.victim, self.mafs, [0]*len(self.victim)], dtype=torch.float32).transpose(0, 1)
        total_snps = self.attacker_state[:, 0].sum().item()
        # print("There are {} SNPs".format(total_snps))
        # return self.attacker_state

    def _init_optimal_attacker(self):
        maf_values = self.beacon_state[0, :, 1]
        sorted_gene_indices = np.argsort(maf_values) # Sort the MAFs 
        print("Victim : {}".format(self.victim))
        print("Victim Sorted Genes: {}".format(sorted_gene_indices))
        mask_one = (self.victim[sorted_gene_indices] == 1) #Get the SNPs
        mask_zero = (self.victim[sorted_gene_indices] == 0) #Get the SNPs
        queries = torch.concatenate((sorted_gene_indices[mask_one], sorted_gene_indices[mask_zero]))[:self.max_steps]
        print(f"Attacker Queries: {queries}")
        return queries
    
    def _calc_control_lrts(self):
        # print(self.args.control_size, len(self.optimal_queries))
        lrts = torch.zeros(size=(self.args.control_size, len(self.optimal_queries)))
        # print(lrts.size())
        for index, ind in enumerate(self.scontrol_state):
            # print("IND", ind)

            sorted_gene_indices = np.argsort(ind[:, 1]) # Sort the MAFs 
            mask_one = (ind[sorted_gene_indices, 0] == 1) #Get the SNPs
            mask_zero = (ind[sorted_gene_indices, 0] == 0) #Get the Non SNPs
            queries = torch.concatenate((sorted_gene_indices[mask_one], sorted_gene_indices[mask_zero]))[:self.max_steps]
            state = copy.deepcopy(self.beacon_state)
            statee = torch.concatenate([state, self.scontrol_state])
            # print("Queries: {}".format(queries))
            # print(statee, statee.size())
            for im, query in enumerate(queries):
                # print(index, im)
                # print("Query: ", query)

                # statee[:, query, 3] = 1
                # agent.policy_old.actor.eval()
                # with torch.no_grad():
                #     response, _, _ = agent.policy_old.act(torch.FloatTensor(torch.flatten(statee)).to(self.args.device))
                # print("response: {}".format(response))


                statee[:, query, 2] = torch.as_tensor(1)

                lrt = self._calc_group_lrts(statee, True)[self.args.beacon_size + index]
                # statee[:, query, 3] = 0
                # print(statee)
                # print(statee)

                # print("LRT ", lrt)

                lrts[index, im] = lrt
                # responses[index, im] = response
            self.control_lrts = lrts
            # self.control_responses = responses
        # agent.policy_old.actor.train()
        return None


    # Initilize the beacon states
    def _init_beacon(self)->None:
        temp_maf = torch.Tensor(self.mafs).unsqueeze(0).expand(self.args.beacon_size, -1)
        responses = torch.zeros(size=(self.args.beacon_size, self.args.gene_size))
        current_query = torch.zeros(size=(self.args.beacon_size, self.args.gene_size))
        lrts = torch.zeros(size=(self.args.beacon_size, self.args.gene_size))

        # print(temp_maf.size(), responses.size(), torch.Tensor(self.s_beacon.T).size())
        self.beacon_state = torch.stack([torch.Tensor(self.s_beacon.T), temp_maf, responses, current_query, lrts], dim=-1)
        # return self.beacon_state

    # Initilize the control group states
    def _init_control(self)->None:
        temp_maf = torch.Tensor(self.mafs).unsqueeze(0).expand(self.args.control_size, -1)
        responses = torch.zeros(size=(self.args.control_size, self.args.gene_size))
        current_query = torch.ones(size=(self.args.control_size, self.args.gene_size))
        lrts = torch.zeros(size=(self.args.control_size, self.args.gene_size))

        # print(temp_maf.size(), responses.size(), torch.Tensor(self.s_beacon.T).size())
        self.scontrol_state = torch.stack([torch.Tensor(self.s_control.T), temp_maf, responses, current_query, lrts], dim=-1)
        # return self.beacon_state

    
    def _calc_group_lrts(self, group, save=True)->float:
        lrt_values = []
        for index, individual in enumerate(group):
            lrts = self.calculate_lrt(group[index, :, :])
            lrt_values.append(torch.sum(lrts))
            if save:
                group[index, :, 4] = torch.Tensor(lrts)
        # min_lrt = min(lrt_values)
        # print("Min LRT: ", min_lrt)
        return torch.Tensor(lrt_values)
    
    def _calc_pvalues(self):
        beacon_lrts = self._calc_group_lrts(self.beacon_state, True)
        control_lrts = self.control_lrts[:, self.current_step]

        print("Victim's LRT: {}".format(beacon_lrts[-1]))
        # print(len(beacon_lrts), len(control_lrts))
        print("Control LRTS: ", self.control_lrts[:, self.current_step])

        pvalues=[]
        for blrt in beacon_lrts:
            pvalue=torch.sum(blrt >= control_lrts) / self.args.control_size
            pvalues.append(pvalue)
        
        print("Beacons min: {} and max: {} LRT".format(torch.min(beacon_lrts), torch.max(beacon_lrts)))
        print("Threshold LRT: {}".format(torch.sort(control_lrts)[0][5]))

        print("---------------------------------------------------------------------")

        # print(pvalues)
        if torch.any(torch.Tensor(pvalues) < 0):
            print(beacon_lrts, control_lrts)
            assert "Wrong p value"

        return torch.Tensor(pvalues)

    # Defining the populations and genes randomly
    def get_populations(self):
        if self.args.control_size + self.args.beacon_size > self.beacon.shape[1]:
            raise Exception("Size of the population is too low!")


        # Prepare index arrays for future use
        genes = np.random.permutation(self.beacon.shape[0])[:self.args.gene_size] # Randomly select gene indexes
        shuffled = np.random.permutation(self.beacon.shape[1]) # Randomly select population indexes


        # Difine different groups of people
        victim_ind = shuffled[0]
        # a_cind = shuffled[1:1+self.args.control_size]
        # s_cind = shuffled[1+self.args.control_size:1+self.args.control_size*2]
        s_cind = shuffled[1:1+self.args.control_size]

        # s_ind = shuffled[80:140]


        if np.random.random() < self.args.victim_prob:
            print("Victim is inside the Beacon!")
            s_ind = shuffled[1+self.args.control_size:self.args.control_size+self.args.beacon_size]
            s_ind = np.append(s_ind, victim_ind)
            s_beacon = self.binary[:, s_ind][genes, :] # Victim inside beacon
        else: 
            print("Victim is NOT inside the Beacon!")
            s_ind = shuffled[1+self.args.control_size:self.args.control_size+self.args.beacon_size+1]
            s_beacon = self.binary[:, s_ind][genes, :]

        # a_control = self.binary[:, a_cind][genes, :]
        s_control = self.binary[:, s_cind][genes, :]
        victim = self.binary[:, victim_ind][genes]
        return s_beacon, s_control, None, victim, self.maf[genes]


    def calculate_lrt(self, ind:torch.Tensor, error=0.001)->float:
        lrts = torch.zeros(self.args.gene_size)

        nsnp_msk = (ind[:, 1] == 0)
        ind[nsnp_msk, 1] = torch.as_tensor(0.001)

        # print(person)
        # filtered_ind = ind[~mask]

        # print(ind) 
        queried_spns_msk = ind[:, 2] == 0 #Responses
        filtered_ind = ind[~queried_spns_msk]

        genome = filtered_ind[:, 0]
        maf = filtered_ind[:, 1]
        response = filtered_ind[:, 2]


        DN_i = (1 - maf).pow(2 * 3) 
        DN_i_1 = (1 - maf).pow(2 * 3 - 2)

        # Genome == 1
        log1 = torch.log(DN_i) - torch.log(0.001 * DN_i_1)
        log2 = torch.log((0.001 * DN_i_1 * (1 - DN_i))) - torch.log(DN_i * (1 - 0.001 * DN_i_1))

        # Genome == 0
        log3 = torch.log(DN_i) - torch.log((1 - 0.001) * DN_i_1)
        log4 = torch.log((1 - 0.001) * DN_i_1 * (1 - DN_i)) - torch.log(DN_i * (1 - DN_i_1 * (1 - 0.001)))

        x_hat_i = (genome * response) + ((1 - genome) * (1 - response))
        lrt = (log1 + log2 * x_hat_i)* genome + (log3 + log4 * x_hat_i) * (1 - genome)
        # print("#############################, ")
        # print(genome, maf, response)
        
        # print("#############################, ", lrt)
        lrts[~queried_spns_msk] = lrt
        # print("lrts, lrts.size()", lrts, lrts.size())
        return lrts

    def _calc_lrt_threshold(self, alpha=0.05):
        control_lrts = self._calc_group_lrts(self.scontrol_state, False)
        threshold = np.percentile(control_lrts, 100 * alpha)
        return threshold
    
        # log tensor data into a CSV file
    def log_beacon_state(self, episode):
        for beacon_idx, beacon_data in enumerate(self.beacon_state):
            for gene_idx, gene_data in enumerate(beacon_data):
                snp, maf, res, current, lrts = gene_data
                self.log_env.writerow([episode, beacon_idx+1, gene_idx+1, snp.detach().cpu().numpy(), maf.detach().cpu().numpy(), res.detach().cpu().numpy(), lrts.detach().cpu().numpy(), self.pvalues.detach().cpu().numpy()[beacon_idx]])
        self.log_env.writerow("--------------------------------------------------------------")
        for beacon_idx, beacon_data in enumerate(self.scontrol_state):
            for gene_idx, gene_data in enumerate(beacon_data):
                snp, maf, res, current, lrts = gene_data
                self.log_env.writerow([episode, beacon_idx+1, gene_idx+1, snp.detach().cpu().numpy(), maf.detach().cpu().numpy(), res.detach().cpu().numpy(), lrts.detach().cpu().numpy(), 0])
        self.log_env.writerow("$$$$$$$$$$$$$$$$$$$$$$")
