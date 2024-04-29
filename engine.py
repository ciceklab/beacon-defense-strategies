from datetime import datetime
import os
import copy
import numpy as np
import matplotlib.pyplot as plt

from utils import plot_individual_rewards, plot_rewards, plot_lrts, plot_lrt_stats

import torch

def train(args:object, env:object, ppo_agent:object):
    print("============================================================================================")
    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    env._calc_control_lrts(agent=ppo_agent)

    ################### Logging ###################
    run_num = 0
    current_num_files = next(os.walk(args.results_dir))[2]
    run_num = len(current_num_files)
    log_f_name = args.results_dir + '/PPO_' + "_log_" + str(run_num) + ".txt"

    print("current logging run number for " + " : ", run_num)
    print("logging at : " + log_f_name)

    ################### checkpointing ###################
    directory = args.results_dir + "/weights"
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = directory + "/PPO_{}.pth".format(run_num)
    print("save checkpoint path : " + checkpoint_path)

    log_f = open(log_f_name,"w+")
    # log_f.write('episode,timestep,reward\n')


    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0


    privacy_erewards = []
    utility_erewards = []
    total_erewards = []

    privacy_rewards=[]
    utility_rewards=[]
    total_rewards=[]

    # training loop
    while i_episode <= args.episodes:
        log_f.write("Episode: {}".format(i_episode))

        print("Episode: ", i_episode)

        state = env.reset()[1]
        # print(state.size())
        current_ep_reward = 0

        current_ep_preward = 0
        current_ep_ureward = 0
        current_lrt_values = []

        beacon_actions = []


        for t in range(1, args.max_queries+1):
            # log_f.write("State: {}".format(state))
            
            # print("State: ", state)
            # select action with policy
            state = torch.flatten(state)
            # print(state.size())
            action = ppo_agent.select_action(state)
            beacon_actions.append(action)

            # print("Beacon Action: ", action)
            state, reward, done, rewards, p_values = env.step(action)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step +=1
            current_ep_reward += reward
            current_ep_preward += rewards[0]
            current_ep_ureward += rewards[1]

            privacy_rewards.append(rewards[0])
            utility_rewards.append(rewards[1])
            total_rewards.append(reward)
            if done:
                break

        print("Beacon Action: {}".format(beacon_actions))

        
        if i_episode % 10 == 0:
            print("\n###########################################\n")
            # print("Episode: ", i_episode)
            print("Episode Reward: ", current_ep_reward)
            print("Episode Privacy Reward: ", current_ep_preward)
            print("Episode Utility Reward: ", current_ep_ureward)
            print("Episode Pvalues: ", p_values)
            print("Min Episode pvalue: ", min(p_values))
            
            plt.plot(np.arange(args.beacon_size), p_values)
            plt.axhline(y=0.05, color='r', linestyle='--')
            plt.show()
            print("Victim Episode pvalue: ", p_values[-1])
            print("\n###########################################\n")


        # update PPO agent
        if i_episode % args.update_freq == 0 and i_episode>0:
            print("updating the agent")
            ppo_agent.update()

            # log_beacon_state average reward till last episode
            log_avg_reward = log_running_reward / log_running_episodes
            log_avg_reward = log_avg_reward

            log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
            log_f.flush()

            log_running_reward = 0
            log_running_episodes = 0

            # print average reward till last episode
            print_avg_reward = print_running_reward / print_running_episodes
            print_avg_reward = print_avg_reward

            print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

            env.log_beacon_state(i_episode)

            print_running_reward = 0
            print_running_episodes = 0
            env._calc_control_lrts(agent=ppo_agent)

            # save model weights
            print("--------------------------------------------------------------------------------------------")
            print("saving model at : " + checkpoint_path)
            ppo_agent.save(checkpoint_path)
            print("model saved")
            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            print("--------------------------------------------------------------------------------------------")
        
        action_std_decay_rate = 0.05    # linearly decay action_std (action_std = action_std - action_std_decay_rate)
        min_action_std = 0.05  

        # # if continuous action space; then decay action std of ouput action distribution
        if i_episode % 50==0 and i_episode>0 == 0:
            ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

        total_erewards.append(current_ep_reward)
        utility_erewards.append(current_ep_ureward)
        privacy_erewards.append(current_ep_preward)
        # lrt_values_list.append(current_lrt_values)

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        if i_episode % args.plot_freq == 0 and i_episode > 0:
            plot_rewards(total_erewards, utility_erewards, privacy_erewards)
            plot_individual_rewards(total_erewards, utility_erewards, privacy_erewards)

            # plot_rewards(total_rewards, utility_rewards, privacy_rewards)
            # plot_individual_rewards(total_rewards, utility_rewards, privacy_rewards)
            # plot_lrt_stats(lrt_values_list)
            # plot_lrts(lrt_values_list)

        if i_episode>0 and i_episode % args.val_freq == 0:
            val(args, ppo_agent=copy.deepcopy(ppo_agent), env=env)

        i_episode += 1

    log_f.close()


    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")

def val(args, ppo_agent, env):
    print("\n=============================================\n")
    print("Start Validating Using Optimal Attacker")
    ppo_agent.policy_old.actor.eval()

    state = env.reset()[1]
    total_reward = 0
    current_ureward = 0
    current_preward = 0
    privacy_rewards = []
    utility_rewards = []
    total_rewards = []
    lrt_values_list = []


    done = False
    while not done:
        state = torch.flatten(state)
        with torch.no_grad():
            action, _, _ = ppo_agent.policy_old.act(state.to(args.device))

        print("Beacon Action: ", action)

        action = action.squeeze().item()
        # rounded_number = round(action, 6)  # Round to 6 decimal places
        #action = action.squeeze().item()
        state, reward, done, rewards, lrt_values = env.step([action])

        total_reward += reward
        current_preward += rewards[0]
        current_ureward += rewards[1]

        print("Current Privacy Reward: {}\nCurrent Utility Reward: {}\nThis Episode Reward: {}\nTotal Reward: {}".format(rewards[0], rewards[1], reward, total_reward))
        total_rewards.append(total_reward)
        utility_rewards.append(current_ureward)
        privacy_rewards.append(current_preward)
        lrt_values_list.append(lrt_values)

    # plot_rewards(total_rewards, utility_rewards, privacy_rewards)
    # plot_individual_rewards(total_reward, utility_rewards, privacy_rewards)
    # plot_lrts(lrt_values_list)
    # plot_lrt_stats(lrt_values_list)
    print(f"Validation completed, total reward")
    print("\n=============================================\n")
    ppo_agent.policy_old.actor.train()
    return total_reward