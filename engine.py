from datetime import datetime
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from ppo import PPO

import torch

def train_beacon(args:object, env:object, ppo_agent:object, attacker_agent=None):
    print("============================================================================================")
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("============================================================================================")

    ################### Logging ###################
    run_num = 0
    current_num_files = next(os.walk(args.results_dir))[2]
    run_num = len(current_num_files)
    print("current logging run number for " + " : ", run_num)

    ################### checkpointing ###################
    directory = args.results_dir + "/weights"
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = directory + "/PPO_{}.pth".format(run_num)
    print("save checkpoint path : " + checkpoint_path)

    time_step = 0
    i_episode = 1

    # training loop
    while i_episode <= args.episodes:
        current_ep_reward = 0
        current_ep_areward = 0
        for t in range(1, args.max_queries+1):
            _, rewards, done, _ = env.step(beacon_agent=ppo_agent, attacker_agent=attacker_agent)

            beacon_reward = rewards[0]

            current_ep_reward += beacon_reward
            current_ep_areward += rewards[1]


            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(beacon_reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step +=1

            if done:
                break


        action_std_decay_rate = 0.05    # linearly decay action_std (action_std = action_std - action_std_decay_rate)
        min_action_std = 0.05  

        # # if continuous action space; then decay action std of ouput action distribution
        if i_episode % 50==0 and i_episode>0 == 0:
            ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

        print("Victim: {} \t Timestep : {} \t Beacon Reward : {}\t Attacker Reward : {}".format(env.victim_id, time_step, current_ep_reward, current_ep_areward))

        # update PPO agent
        if i_episode % args.update_freq == 0 and i_episode>0:
            print("--------------------------------------------------------------------------------------------")
            print("updating the agent")
            ppo_agent.update()

            # save model weights
            print("saving model at : " + checkpoint_path)
            ppo_agent.save(checkpoint_path)
            print("model saved")
            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            print("--------------------------------------------------------------------------------------------")
        

        i_episode += 1
        env.reset()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")




def train_attacker(args:object, env:object, ppo_agent:object):
    print("============================================================================================")
    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    ################### Logging ###################
    run_num = 0
    current_num_files = next(os.walk(args.results_dir))[2]
    run_num = len(current_num_files)
    # print("current logging run number for " + " : ", run_num)

    ################### checkpointing ###################
    directory = args.results_dir + "/weights"
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = directory + "/PPO_{}.pth".format(run_num)
    # print("save checkpoint path : " + checkpoint_path)

    time_step = 0
    i_episode = 1

    # training loop
    while i_episode <= args.episodes:
        current_ep_reward = 0
        for t in range(1, args.max_queries+1):
            _, rewards, done, _ = env.step(attacker_agent=ppo_agent)
            # print(rewards)
            attacker_reward = rewards[1]
            current_ep_reward += attacker_reward
            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(attacker_reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step +=1

            if done:
                break

        print("Victim: {} \t Timestep : {} \t Current Episode Reward : {}".format(env.victim_id, time_step, current_ep_reward))

        # update PPO agent
        if i_episode % args.update_freq == 0 and i_episode>0:
            print("--------------------------------------------------------------------------------------------")
            print("updating the agent")
            ppo_agent.update()
            time_step = 0

            # save model weights
            print("saving model at : " + checkpoint_path)
            ppo_agent.save(checkpoint_path)
            print("model saved")
            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            print("--------------------------------------------------------------------------------------------")

        i_episode += 1

        env.reset()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")



def train_both(args:object, env:object, beacon_agent:object, attacker_agent=None):
    print("============================================================================================")
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("============================================================================================")

    ################### Logging ###################
    run_num = 0
    current_num_files = next(os.walk(args.results_dir))[2]
    run_num = len(current_num_files)
    print("current logging run number for " + " : ", run_num)

    ################### checkpointing ###################
    directory = args.results_dir + "/weights"
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path_beacon = directory
    checkpoint_path_attacker = directory + "/PPO_attacker{}.pth".format(run_num)

    print("save checkpoint path : " + checkpoint_path_beacon)

    time_step = 0
    i_episode = 1

    # training loop
    while i_episode <= args.episodes:
        current_ep_reward = 0
        current_ep_areward = 0
        for t in range(1, args.max_queries+1):
            binfo, rewards, done, _ = env.step(beacon_agent=beacon_agent, attacker_agent=attacker_agent)

            beacon_reward = rewards[0]
            attacker_reward = rewards[1]

            current_ep_reward += beacon_reward
            current_ep_areward += rewards[1]

            # saving reward and is_terminals 
            if args.beacon_agent == "ppo":
                beacon_agent.buffer.rewards.append(beacon_reward)
                beacon_agent.buffer.is_terminals.append(done)
            elif args.beacon_agent == "td":
                beacon_agent.buffer.store(binfo[0], binfo[1], binfo[2], binfo[3], binfo[4],)
            else:
                raise NotImplemented    

            attacker_agent.buffer.rewards.append(attacker_reward)
            attacker_agent.buffer.is_terminals.append(done)

            time_step +=1

            if done:
                break

        if args.beacon_agent == "ppo":
            action_std_decay_rate = 0.05    # linearly decay action_std (action_std = action_std - action_std_decay_rate)
            min_action_std = 0.05  

            # if continuous action space; then decay action std of ouput action distribution
            if i_episode % 50==0 and i_episode>0 == 0:
                beacon_agent.decay_action_std(action_std_decay_rate, min_action_std)

        print("Victim: {} \t Timestep : {} \t Beacon Reward : {}\t Attacker Reward : {}".format(env.victim_id, time_step, current_ep_reward, current_ep_areward))

        # update PPO agent
        if i_episode % args.update_freq == 0 and i_episode>100:
            print("--------------------------------------------------------------------------------------------")
            print("updating the agent")
            beacon_agent.update()
            attacker_agent.update()
            beacon_agent.explore_noise *= 0.9998

            # save model weights
            print("saving model at : " + checkpoint_path_beacon)
            beacon_agent.save(checkpoint_path_beacon)
            attacker_agent.save(checkpoint_path_attacker)
            print("model saved")
            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            print("--------------------------------------------------------------------------------------------")
        

        i_episode += 1
        env.reset()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")



def train_TD_beacon(args:object, env:object, ppo_agent:object, attacker_agent=None):
    print("============================================================================================")
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("============================================================================================")

    ################### Logging ###################
    run_num = 0
    current_num_files = next(os.walk(args.results_dir))[2]
    run_num = len(current_num_files)
    print("current logging run number for " + " : ", run_num)

    ################### checkpointing ###################
    directory = args.results_dir + "/weights"
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = directory
    print("save checkpoint path : " + checkpoint_path)

    time_step = 1
    i_episode = 1

    # training loop
    while i_episode <= args.episodes:
        breaked=False
        current_ep_reward = 0
        current_ep_areward = 0

        binfo, rewards, done, _ = env.step(beacon_agent=ppo_agent, attacker_agent=attacker_agent)
        state, action, reward, dw = binfo

        for t in range(1, args.max_queries+1):
            
            binfo, rewards, done, _ = env.step(beacon_agent=ppo_agent, attacker_agent=attacker_agent)
            # print("Buffer: ", state, action, reward, binfo[0], dw)
            ppo_agent.buffer.store(state, action, reward, binfo[0], dw)

            state, action, reward, dw = binfo

            beacon_reward = rewards[0]

            current_ep_reward += beacon_reward
            current_ep_areward += rewards[1]

            time_step +=1
            
            if breaked:
                break

            if done:
                breaked=True

        print("Victim: {} \t Timestep : {} \t Beacon Reward : {}\t Attacker Reward : {}".format(env.victim_id, time_step, current_ep_reward, current_ep_areward))

        # update PPO agent
        if i_episode % args.update_freq == 0 and i_episode>100:
            print("--------------------------------------------------------------------------------------------")
            print("updating the agent")
            ppo_agent.update()
            ppo_agent.explore_noise *= 0.9998

            # save model weights
            print("saving model at : " + checkpoint_path)
            ppo_agent.save(checkpoint_path)
            print("model saved")
            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            print("--------------------------------------------------------------------------------------------")
        

        i_episode += 1
        env.reset()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")

