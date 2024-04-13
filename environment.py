import numpy as np
import copy
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
        
        # print(self.attacker_state.size())
        # print(self.beacon_state.size())


        # Define action and observation space
        self.action_space = spaces.Box(low=0, high=1, shape=(1,))  # Beacon lies or not
        self.observation_space = spaces.Box(low=-1, high=1,shape=(args.beacon_size, args.gene_size, 4))  # State: [Beacon Size, Gene size, 3(SNP, MAF, RES, CURRENT)]
        self.max_steps = args.max_queries  # Maximum number of steps per episode
        self.current_step = 0

    def start(self):
        self.attacker_action = np.random.randint(low=0, high=self.args.gene_size)
        current_beacon_s = copy.deepcopy(self.beacon_state)
        current_beacon_s[:, self.attacker_action, 3] = 1
        return current_beacon_s, 0, False, {}

    # Reset the environment after an episode
    def reset(self) -> torch.Tensor:
        self.reset_counter+=1
        if self.reset_counter==self.args.pop_reset_freq:
            print("Reseting the Populations")
            self._reset_populations()
            self.reset_counter = 0   


        # Reset the states of our agents
        self._init_attacker()
        self._init_beacon()

        self.altered_probs = 0
        
        self.current_step = 0
        return self.attacker_state, self.beacon_state

    def step(self, beacon_action): 
        beacon_action = np.clip(beacon_action, -1, 1)
        done = False
        # # Change the res of the asked gene to 1 in the state of beacon
        # if beacon_action > 0.5:
        self.beacon_state[:, self.attacker_action, 2] += beacon_action #TRUTH
        # else:
        #     self.beacon_state[:, self.attacker_action, 2] = -1 #LIE

        # Take attacker Action
        self.attacker_action = np.random.randint(low=0, high=self.args.gene_size)
        current_beacon_s = copy.deepcopy(self.beacon_state)
        current_beacon_s[:, self.attacker_action, 3] = 1

        observation = current_beacon_s # Attacker's state will be added for the multi agent RL


        # Calculate the lrt for individuals in the beacon and find the min 
        self.altered_probs += beacon_action
        preward = torch.exp(self._calc_beacon_reward())
        ureward = -1*self.altered_probs
        print("preward: ", preward)
        print("ureward: ", ureward)
        reward = preward + ureward

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        
        return observation, reward, done, [preward, ureward]

    def _reset_populations(self)->None:
        self.s_beacon, self.s_control, self.a_control, self.victim, self.mafs = self.get_populations()

            # Initilize the attacker states
    def _init_attacker(self)->None:
        self.attacker_state = torch.tensor([self.victim, self.mafs, [0]*len(self.victim)], dtype=torch.float32).transpose(0, 1)
        total_snps = self.attacker_state[:, 0].sum().item()
        # print("There are {} SNPs".format(total_snps))
        # return self.attacker_state

    # Initilize the beacon states
    def _init_beacon(self)->None:
        temp_maf = torch.Tensor(self.mafs).unsqueeze(0).expand(self.args.beacon_size, -1)
        responses = torch.zeros(size=(self.args.beacon_size, self.args.gene_size))
        current_query = torch.zeros(size=(self.args.beacon_size, self.args.gene_size))
        # print(temp_maf.size(), responses.size(), torch.Tensor(self.s_beacon.T).size())
        self.beacon_state = torch.stack([torch.Tensor(self.s_beacon.T), temp_maf, responses, current_query], dim=-1)
        # return self.beacon_state
    
    def _calc_beacon_reward(self):
        lrt_values = []
        for index, individual in enumerate(self.beacon_state):
            lrt = self.calculate_lrt(self.beacon_state[index, :, :])
            lrt_values.append(lrt)
            # print("lrt_values", lrt_values)
        min_lrt = min(lrt_values)
        # print("Min LRT: ", min_lrt)
        return min_lrt

    # Defining the populations and genes randomly
    def get_populations(self):
        if self.args.control_size*2 + self.args.beacon_size > self.beacon.shape[1]:
            raise Exception("Size of the population in too low!")


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


    def calculate_lrt(self, ind, error=0.001):
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
        log1 = torch.log(DN_i / (error * DN_i_1))
        log2 = torch.log((error * DN_i_1 * (1 - DN_i)) / (DN_i * (1 - error * DN_i_1)))

        # Genome == 0
        log3 = torch.log(DN_i / ((1 - error) * DN_i_1))
        log4 = torch.log((1 - error) * DN_i_1 * (1 - DN_i)) / (DN_i * (1 - DN_i_1 * (1 - error)))

        x_hat_i = (genome * response) + ((1 - genome) * (1 - response))

        lrts = (log1 + log2 * x_hat_i) * genome + (log3 + log4 * x_hat_i) * (1 - genome)

        nan_mask = torch.isnan(lrts)
        lrts = lrts[~nan_mask]
        return torch.sum(lrts)