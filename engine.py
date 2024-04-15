from datetime import datetime
import os
import copy

from utils import plot_individual_rewards, plot_rewards, plot_lrts

import torch

def train(args:object, env:object, ppo_agent:object):
    print("============================================================================================")
    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")


    ################### Logging ###################
    run_num = 0
    current_num_files = next(os.walk(args.results_dir))[2]
    run_num = len(current_num_files)
    log_f_name = args.results_dir + '/PPO_' + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + " : ", run_num)
    print("logging at : " + log_f_name)

    ################### checkpointing ###################
    directory = args.results_dir + "/weights"
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = directory + "/PPO_{}.pth".format(run_num)
    print("save checkpoint path : " + checkpoint_path)

    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')


    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0


    privacy_rewards = []
    utility_rewards = []
    total_rewards = []
    lrt_values_list = []


    # training loop
    while i_episode <= args.episodes:

        state = env.reset("random")[1]
        # print(state.size())
        current_ep_reward = 0

        current_ep_preward = 0
        current_ep_ureward = 0
        current_lrt_values = []

        print("Episode: ", i_episode)


        for t in range(1, args.max_queries+1):

            # select action with policy
            state = torch.flatten(state)
            action = ppo_agent.select_action(state)
            state, reward, done, rewards, lrt_values = env.step(action, "random")

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step +=1
            current_ep_reward += reward
            current_ep_preward += rewards[0]
            current_ep_ureward += rewards[1]
            current_lrt_values = lrt_values
            if done:
                break

        # update PPO agent
        if i_episode % (args.pop_reset_freq/2) == 0 and i_episode>0:
            print("updating the agent")
            ppo_agent.update()

            # log average reward till last episode
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

            print_running_reward = 0
            print_running_episodes = 0

            # save model weights
            print("--------------------------------------------------------------------------------------------")
            print("saving model at : " + checkpoint_path)
            ppo_agent.save(checkpoint_path)
            print("model saved")
            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            print("--------------------------------------------------------------------------------------------")


        # # if continuous action space; then decay action std of ouput action distribution
        # if has_continuous_action_space and i_episode % (args.episodes/4)==0 and i_episode>0 == 0:
        #     ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)


        total_rewards.append(current_ep_reward)
        utility_rewards.append(current_ep_ureward)
        privacy_rewards.append(current_ep_preward)
        lrt_values_list.append(current_lrt_values)

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        if i_episode % 50 == 0 and i_episode > 0:
            plot_rewards(total_rewards, utility_rewards, privacy_rewards)
            plot_individual_rewards(total_rewards, utility_rewards, privacy_rewards)
            plot_lrts(lrt_values_list)

        # if i_episode % args.val_freq == 0:
        #     val(args, ppo_agent=copy.deepcopy(ppo_agent), env=env)

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
    print("Start Validating Using Optimal Attacker")

    state = env.reset("")[1]
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
        action, _, _ = ppo_agent.policy.act(state.to(args.device))
        print("arda", action)
        state, reward, done, rewards, lrt_values = env.step(action.detach().cpu().numpy(), "")

        total_reward += reward
        current_preward += rewards[0]
        current_ureward += rewards[1]

        total_rewards.append(total_reward)
        utility_rewards.append(current_ureward)
        privacy_rewards.append(current_preward)
        lrt_values_list.append(lrt_values)

        # print("total reward ", reward, " preward: ", rewards[0], " ureward: ", rewards[1])

    print(total_rewards, utility_rewards, privacy_rewards, lrt_values)
    plot_rewards(total_rewards, utility_rewards, privacy_rewards)
    #plot_individual_rewards(total_reward, utility_rewards, privacy_rewards)
    plot_lrts(lrt_values_list)
    print(f"Validation completed, total reward: {total_reward}")
    return total_reward


# Validation
'''def val(ppo_agent, env):

    #TODO: get the actor of ppo agent first eval the model to prevent changing the model

    action = ppo_agent.actor.eval()
    state, reward, done, rewards = env.step(action)

    print("total reward ", reward, " preward: ", rewards[0], " ureward: ", rewards[1])


    # Send Optimal Attacker quieries to the env get the state of beacon
    # Act according to the state using actor model and log the rewards and actions
    return None
'''