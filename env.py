import numpy as np
import copy
import os
import csv
import math
import random

from utils import plot_lists, lrt, create_csv, log_env, log_victim, plot_two_lists, plot_three_lists
from beacon import Beacon
from attacker import Attacker

import torch

class Env():
    def __init__(self, args, beacon, maf, binary):

        self.beacon_people=beacon
        self.maf=maf
        self.args=copy.copy(args)
        self.binary=binary

        ############################LOG

        self.log_beacon = create_csv(self.args.results_dir, "beacon_log")
        self.log_beacon_control = create_csv(self.args.results_dir, "beacon_control_log")

        self.log_attacker = create_csv(self.args.results_dir, "attacker_log")
        self.log_attacker_control = create_csv(self.args.results_dir, "attacker_control_log")


        self.episode = 0
        print("==================================⬇️⬇️⬇️⬇️Episode: {}⬇️⬇️⬇️⬇️==============================".format(self.episode+1))


        # Randomly set populations and genes
        beacon_case, beacon_control, attack_control, self.victim, self.victim_id, mafs = self.get_populations()

        # Initialize the agents
        self.beacon = Beacon(self.args, beacon_case, beacon_control, mafs, self.victim_id)
        self.attacker = Attacker(self.args, self.victim, attack_control, mafs)
        # print("-------------------------")

        self.attacker_action = 0
        self.current_step = 0
        self.altered_probs = 0
        self.max_steps = self.args.max_queries  # Maximum number of steps per episode

        self.beacon_prewards = []
        self.beacon_urewards = []
        self.beacon_total = []


        self.attacker_prewards = []
        self.attacker_urewards = []
        self.attacker_total = []

        self.attacker_actions = []

        self.attacker_agent_actions = []
        self.beacon_agent_actions = []


   #################################################################################################
    #RESET STEP

    # Reset the environment after an episode
    def reset(self) -> torch.Tensor:
        self.episode += 1

        self.beacon_prewards = []
        self.beacon_urewards = []
        self.beacon_total = []
        self.attacker_urewards = []
        self.attacker_prewards = []
        self.attacker_total = []
        self.attacker_actions = []

        self.attacker_agent_actions = []
        self.beacon_agent_actions = []

        print("==================================⬇️⬇️⬇️⬇️Episode: {}⬇️⬇️⬇️⬇️==============================".format(self.episode+1))

        # print("Reseting the Populations")
        # Randomly set populations and genes
        beacon_case, beacon_control, attack_control, victim, self.victim_id, mafs = self.get_populations()


        self.altered_probs = 0
        self.current_step = 0
        self.lie_probs = 0


        # Reset the agents 
        self.attacker = Attacker(args=self.args, victim=victim, control=attack_control, mafs=mafs)
        self.beacon = Beacon(self.args, beacon_case, beacon_control, mafs, self.victim_id)



    def step(self, beacon_agent=None, attacker_agent=None):
        # print("Query Number: {}".format(self.current_step))
        done = False
        # print("--------------------------------States---------------------------------")

        ################# Take the actions
        if self.args.attacker_type == "agent":
            attacker_state = self.attacker.get_state()
            attacker_state = torch.flatten(attacker_state).float()
            # print("Attacker State: ", attacker_state)
            agent_action = attacker_agent.select_action(attacker_state) 
            attacker_action = self.attacker.act(self.current_step, agent_action) 
            # print("Attacker Action: {}, {}".format(attacker_action, agent_action))
        else:
            attacker_state = self.attacker.get_state()
            attacker_action = self.attacker.act(self.current_step) 
        
        if self.args.beacon_type == "agent":
            beacon_state = self.beacon.get_state(attacker_action) 
            beacon_action = beacon_agent.select_action(beacon_state, False) 
            beacon_action = torch.Tensor(beacon_action)

        else:
            beacon_state = self.beacon.get_state(attacker_action) 
            beacon_action = self.beacon.act() 

        self.attacker_agent_actions.append(agent_action)
        self.beacon_agent_actions.append(beacon_action)

        ################# Save the Actions
        # beacon_action = torch.clamp(torch.as_tensor(beacon_action), min=0, max=1)

        # print("--------------------------------Actions---------------------------------")
        # print("Attacker Action: Position {} with MAF: {} and SNP: {} and LRT: {}".format(attacker_action, self.maf[attacker_action], self.victim[attacker_action], lrt(number_of_people=self.args.beacon_size, genome=self.victim[attacker_action], maf=self.maf[attacker_action], response=beacon_action)))
        # print("Beacon Action: {}".format(beacon_action))
        # print("-----------------------------------------------------------------")

        # print("--------------------------------Updating---------------------------------")

        ########## Update the states
        self.beacon.update(beacon_action=beacon_action, attacker_action=attacker_action)
        self.attacker.update(beacon_action=beacon_action, attacker_action=attacker_action)

        # print("--------------------------------Rewards---------------------------------")

        ########### Calculate Rewards
        beacon_reward, beacon_done, beacon_rewards = self.beacon.calc_reward(beacon_action=beacon_action)
        attacker_reward, attacker_done, attacker_rewards = self.attacker.calc_reward(beacon_action, self.current_step)

        # print("Beacon utility reward: {}  privacy reward: {}".format(beacon_rewards[1], beacon_rewards[0]))

        if attacker_action in self.attacker_actions:
            attacker_reward -= 5
        

        self.beacon_prewards.append(beacon_rewards[0])
        self.beacon_urewards.append(beacon_rewards[1])
        self.beacon_total.append(beacon_reward)
        self.attacker_prewards.append(attacker_rewards[0])
        self.attacker_urewards.append(attacker_rewards[1])
        self.attacker_total.append(attacker_reward)
        self.attacker_actions.append(attacker_action)

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
            print("✅✅✅ Attacker Could NOT Indentify the VICTIM ✅✅✅")

            
        done = beacon_done or done or attacker_done

        log_env(info=self.beacon.beacon_info, episode=self.episode, step=self.current_step, log_env_name=self.log_beacon)
        log_env(info=self.beacon.control_info, episode=self.episode, step=self.current_step, log_env_name=self.log_beacon_control)

        log_victim(info=self.attacker.victim_info, episode=self.episode, step=self.current_step, log_env_name=self.log_attacker)
        log_env(info=self.attacker.control_info, episode=self.episode, step=self.current_step, log_env_name=self.log_attacker_control)


        if done:
            self.beacon.get_state(self.attacker_action)
            self.attacker.get_state()

        if done and (self.episode % 100 == 0):
            plot_two_lists(self.beacon_agent_actions, self.attacker_agent_actions, path=self.args.results_dir + "/actions", name="action", episode=self.episode, xlabel="Episode 10th", ylabel='Actions')
            plot_three_lists(self.beacon_prewards, self.beacon_urewards, self.beacon_total, path=self.args.results_dir + "/rewards", name="beacon", label1="Beacon Privacy Reward", label2="Beacon Utility Reward", label3="Total", episode=self.episode, xlabel="Episode 10th", ylabel='Beacon Rewards')
            plot_three_lists(self.attacker_prewards, self.attacker_urewards, self.attacker_total, path=self.args.results_dir + "/rewards", label1="Attacker -1 Reward", label2="Attacker beacon action Reward", label3="Total", name="attacker", episode=self.episode, xlabel="Episode 10th", ylabel='Attacker Rewards')



        # if self.episode % self.args.plot_freq == 0:
        #     plot_lists(values=self.beacon_prewards, path=self.args.results_dir + "/indrewards", name="b_preward", episode=self.episode, thresh=0.05)
        #     plot_lists(values=self.beacon_urewards, path=self.args.results_dir + "/indrewards", name="b_ureward", episode=self.episode, thresh=0.05)
        #     plot_lists(values=self.attacker_prewards, path=self.args.results_dir + "/indrewards", name="a_preward", episode=self.episode)
        #     plot_lists(values=self.attacker_urewards, path=self.args.results_dir + "/indrewards", name="ae_preward", episode=self.episode)

            
            # plot_lists()
            # plot_lists()


        return [np.array(beacon_state), beacon_action, beacon_reward, np.array(self.beacon.get_state(self.attacker_action)), done], [beacon_reward, attacker_reward], done, []
    
    #################################################################################################
    #Populations

    # Defining the populations and genes randomly
    def get_populations(self):
        print("-------------------------")

        if 1 + self.args.a_control_size + self.args.b_control_size + self.args.beacon_size > self.beacon_people.shape[1]:
            raise Exception("Size of the population is too low!")
        
        beacon_size = self.args.beacon_size + 1
        all_people = self.binary[:, :self.args.gene_size]

        # Beacon Case group
        beacon_case = all_people[:beacon_size]
        # Define control groups
        beacon_control = all_people[beacon_size:beacon_size + self.args.b_control_size]
        attack_control = all_people[beacon_size + self.args.b_control_size:beacon_size + self.args.b_control_size + self.args.a_control_size]

        victim_ind = np.random.randint(1, beacon_size)
        # victim_ind = self.episode +1 
    
        victim = beacon_case[victim_ind]

        if np.random.random() < self.args.victim_prob:
            # print("Victim is inside the Beacon!")
            # print(f"Victim is the {victim_ind-1}th person")
            beacon_case = np.delete(beacon_case, 0, axis=0)
        else:
            # print("Victim is NOT inside the Beacon!")
            beacon_case = np.delete(beacon_case, victim_ind, axis=0)

        victim_ind -= 1
        # print("-------------------------")

        # Just for test 
        beacon_control = attack_control
        # print(beacon_case)
        return torch.Tensor(beacon_case), torch.Tensor(beacon_control), torch.Tensor(attack_control), torch.Tensor(victim), victim_ind, torch.Tensor(self.maf[:self.args.gene_size])
    


