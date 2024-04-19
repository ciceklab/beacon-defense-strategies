import numpy as np
import copy
import os
import csv
import math

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
        self.reset_counter = -1

        # Initialize the agents
        self._init_attacker()
        self._init_beacon()


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


        log_env_name = args.results_dir + "/env" + '/PPO_' + str(len(next(os.walk(args.results_dir))[2])) + ".csv"
        self.log_env = csv.writer(open(log_env_name,"w+"))
        self.log_env.writerow(["Episode", "Beacon", "Gene", "SNP", "MAF", "RES"])
        # log_f.write('Start\n')



    # Reset the environment after an episode
    def reset(self) -> torch.Tensor:
        self.reset_counter+=1

        if self.reset_counter%self.args.pop_reset_freq==0 and self.reset_counter>0:
            print("Reseting the Populations")
            self._reset_populations()
            self.reset_counter = 0   


        # Reset the states of our agents
        self._init_attacker()
        self._init_beacon()

        self.altered_probs = 0
        self.current_step = 0

        if self.reset_counter==0:
            if self.args.attacker_type == "random":
                self.attacker_action = np.random.randint(low=0, high=self.args.gene_size)
            elif self.args.attacker_type == "optimal":
                self.attacker_action = self.optimal_queries[self.current_step]
            else:
                raise NotImplemented
            current_beacon_s = copy.deepcopy(self.beacon_state)
            current_beacon_s[:, self.attacker_action, 3] = 1
            return self.attacker_state, current_beacon_s


        return self.attacker_state, self.beacon_state

    def step(self, beacon_action:float):
        beacon_action = np.clip(beacon_action, 0, 1)
        done = False
        # # Change the res of the asked gene to 1 in the state of beacon
        # if beacon_action > 0.5:
        self.beacon_state[:, self.attacker_action, 2] = torch.Tensor(beacon_action) #TRUTH
        # else:
        #     self.beacon_state[:, self.attacker_action, 2] = -1 #LIE

        # Take attacker Action
        if self.args.attacker_type == "random":
            self.attacker_action = np.random.randint(low=0, high=self.args.gene_size)
        elif self.args.attacker_type == "optimal":
            self.attacker_action = self.optimal_queries[self.current_step]
        else:
            raise NotImplemented
        
        current_beacon_s = copy.deepcopy(self.beacon_state)
        current_beacon_s[:, self.attacker_action, 3] = 1
        observation = current_beacon_s # Attacker's state will be added for the multi agent RL


        # Calculate the lrt for individuals in the beacon and find the min 
        self.altered_probs += beacon_action
        min_lrt, lrt_values = self._calc_beacon_reward()
        preward = min_lrt
        ureward = self.altered_probs
        # print("lrt: ", self._calc_beacon_reward())
        # print("preward: ", preward)
        # print("ureward: ", ureward)
        reward = preward + ureward

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
        
        if self.current_step == len(self.optimal_queries)-1:
            done = True

        return observation, reward, done, [preward, ureward], lrt_values

    def _reset_populations(self)->None:
        self.s_beacon, self.s_control, self.a_control, self.victim, self.mafs = self.get_populations()

            # Initilize the attacker states
    def _init_attacker(self)->None:
        self.attacker_state = torch.tensor([self.victim, self.mafs, [0]*len(self.victim)], dtype=torch.float32).transpose(0, 1)
        total_snps = self.attacker_state[:, 0].sum().item()
        # print("There are {} SNPs".format(total_snps))
        # return self.attacker_state

    def _init_optimal_attacker(self):
        maf_values = self.beacon_state[0, :, 1]
        sorted_gene_indices = np.argsort(maf_values) # Sort the MAFs 
        mask = (self.victim[sorted_gene_indices] == 1) #Get the SNPs
        return sorted_gene_indices[mask]
    

    # Initilize the beacon states
    def _init_beacon(self)->None:
        temp_maf = torch.Tensor(self.mafs).unsqueeze(0).expand(self.args.beacon_size, -1)
        responses = torch.zeros(size=(self.args.beacon_size, self.args.gene_size))
        current_query = torch.zeros(size=(self.args.beacon_size, self.args.gene_size))
        # print(temp_maf.size(), responses.size(), torch.Tensor(self.s_beacon.T).size())
        self.beacon_state = torch.stack([torch.Tensor(self.s_beacon.T), temp_maf, responses, current_query], dim=-1)
        # return self.beacon_state
    
    def _calc_beacon_reward(self)->float:
        lrt_values = []
        for index, individual in enumerate(self.beacon_state):
            lrt = self.calculate_lrt(self.beacon_state[index, :, :])
            lrt_values.append(lrt)
            # print("lrt_values", lrt_values)
        min_lrt = min(lrt_values)
        # print("Min LRT: ", min_lrt)
        return min_lrt, lrt_values

    # Defining the populations and genes randomly
    def get_populations(self):
        if self.args.control_size*2 + self.args.beacon_size > self.beacon.shape[1]:
            raise Exception("Size of the population is too low!")


        # Prepare index arrays for future use
        genes = np.random.permutation(self.beacon.shape[0])[:self.args.gene_size] # Randomly select gene indexes
        shuffled = np.random.permutation(self.beacon.shape[1]) # Randomly select population indexes


        # Difine different groups of people
        victim_ind = shuffled[0]
        a_cind = shuffled[1:1+self.args.control_size]
        s_cind = shuffled[1+self.args.control_size:1+self.args.control_size*2]
        # s_ind = shuffled[80:140]


        if np.random.random() < self.args.victim_prob:
            print("Victim is inside the Beacon!")
            s_ind = shuffled[1+self.args.control_size*2:self.args.control_size*2+self.args.beacon_size]
            s_ind = np.append(s_ind, victim_ind)
            s_beacon = self.binary[:, s_ind][genes, :] # Victim inside beacon
        else: 
            print("Victim is NOT inside the Beacon!")
            s_ind = shuffled[1+self.args.control_size*2:self.args.control_size*2+self.args.beacon_size+1]
            s_beacon = self.binary[:, s_ind][genes, :]

        a_control = self.binary[:, a_cind][genes, :]
        s_control = self.binary[:, s_cind][genes, :]
        victim = self.binary[:, victim_ind][genes]
        return s_beacon, s_control, a_control, victim, self.maf[genes]


    def calculate_lrt(self, ind:int, error=0.001)->float:
        # print(ind)
        mask = (ind[:, 2] == 0) & (ind[:, 3] == 0)
        filtered_ind = ind[~mask]

        genome = filtered_ind[:, 0]
        maf = filtered_ind[:, 1]
        response = filtered_ind[:, 2]
        # print(genome, maf, response)
        
        DN_i = (1 - maf).pow(2 * self.args.beacon_size) 
        DN_i_1 = (1 - maf).pow(2 * self.args.beacon_size - 2)

        # Genome == 1
        log1 = torch.log(DN_i) - torch.log(error * DN_i_1)
        log2 = torch.log((error * DN_i_1 * (1 - DN_i))) - torch.log(DN_i * (1 - error * DN_i_1))

        # Genome == 0
        log3 = torch.log(DN_i) - torch.log((1 - error) * DN_i_1)
        log4 = torch.log((1 - error) * DN_i_1 * (1 - DN_i)) - torch.log(DN_i * (1 - DN_i_1 * (1 - error)))

        x_hat_i = (genome * response) + ((1 - genome) * (1 - response))

        lrts = (log1 + log2 * x_hat_i) * genome + (log3 + log4 * x_hat_i) * (1 - genome)

        nan_mask = torch.isnan(lrts)
        lrts = lrts[~nan_mask]
        return torch.sum(lrts)
    
        # log tensor data into a CSV file
    def log_beacon_state(self, episode):
        for beacon_idx, beacon_data in enumerate(self.beacon_state):
            for gene_idx, gene_data in enumerate(beacon_data):
                snp, maf, res, current = gene_data
                self.log_env.writerow([episode, beacon_idx+1, gene_idx+1, snp.detach().cpu().numpy(), maf.detach().cpu().numpy(), res.detach().cpu().numpy()])