import matplotlib.pyplot as plt


def plot_rewards(losses1, losses2, losses3):
    epochs = range(1, len(losses1) + 1) 

    plt.figure(figsize=(10, 6))

    plt.plot(epochs, losses1, 'r', label='Total Rewards')
    plt.plot(epochs, losses2, 'g', label='Utility Rewards')
    plt.plot(epochs, losses3, 'b', label='Privacy rewards')

    plt.title('Rewards of model')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()

    plt.show()



def plot_individual_rewards(losses1, losses2, losses3):
    epochs = range(1, len(losses1) + 1)  # Assuming all lists have the same length

    plt.figure(figsize=(10, 6))

    plt.subplot(3, 1, 1)
    plt.plot(epochs, losses1, 'r')
    plt.title('Total Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')

    plt.subplot(3, 1, 2)
    plt.plot(epochs, losses2, 'g')
    plt.title('Utility Loss')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')

    plt.subplot(3, 1, 3)
    plt.plot(epochs, losses3, 'b')
    plt.title('Privacy Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')

    plt.show()


def plot_lrts(lrt_values_list, group_size=10):

    num_individuals = len(lrt_values_list[0])
    num_plots = (num_individuals + group_size - 1) // group_size  #  ensure all individuals are used

    # Create a figure with subplots
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots), constrained_layout=True)
    # Handle the case when there is only one subplot
    '''if num_plots == 1:
        axes = [axes]'''
    #For each plot
    for i in range(num_plots):
        start_index = i * group_size
        end_index = min(start_index + group_size, num_individuals)

        # For each individual
        for individual_index in range(start_index, end_index):
            # Extract this individual's LRT values from all time points
            individual_lrt_values = [time_point[individual_index] for time_point in lrt_values_list]

            axes[i].plot(individual_lrt_values, marker='o', linestyle='-', label=f'Individual {individual_index + 1}')

        axes[i].set_title(f'LRT Values for Individuals {start_index + 1} to {end_index}')
        axes[i].set_xlabel('Array Index')
        axes[i].set_ylabel('LRT Value')
        axes[i].grid(True)
        axes[i].legend()

    plt.show()
