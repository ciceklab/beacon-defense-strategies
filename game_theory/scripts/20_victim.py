import os
import time
import argparse
import warnings
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import permutations, product

from concurrent.futures import ThreadPoolExecutor

from common.data import get_maf_values, pre_process_maf, get_data
from common.scenario import optimal_scenario
from common.utility import UtilityAttacker, UtilityBeacon

warnings.filterwarnings('ignore')

# TODO: load these from cmd params
NUM_QUERIES = 4
POSSBILE_SNPS = 4
EXIST_COUNT = 4
FIGURES_SAVE_DIR = "/results"

def args_create():
    # @title Arguments
    parser = argparse.ArgumentParser(description='Actor Critic')

    # IO
    parser.add_argument('--results_dir', default='./results', type=str, help='Save dir')
    parser.add_argument('--data_path', default="./data", type=str, help='Dataset Path')
    
    # Simulation
    parser.add_argument('--num_queries', default=4, type=int, metavar='N', help='Maximum number of attacker queries.')
    parser.add_argument('--victim_snps', default=4, type=int, help='Number of the SNPs in the beacon that exists in the victim')
    parser.add_argument('--seed', default=2024, type=int, help='Seed for reproducibility')
    parser.add_argument('--a_control_size', default=20, type=int, help='Attack Control group size')
    parser.add_argument('--gene_size', default=100000, type=int, help='Gene size')
    parser.add_argument('--beacon_size', default=10, type=int, help='Beacon population size')
    parser.add_argument('--victim_count', default=20, type=int, help='Number of victims in the simulation')
    parser.add_argument('--victims_in_beacon_count', default=20, type=int, help='Number of victims in the beacon.')
    args = parser.parse_args() 

    args.results_dir = os.path.join(args.results_dir, "run"+str(len(os.listdir(args.results_dir))))
    os.makedirs(args.results_dir)
    os.makedirs(args.results_dir+"/logs")
    os.makedirs(args.results_dir+"/rewards")
    os.makedirs(args.results_dir+"/indrewards")
    os.makedirs(args.results_dir+"/actions")
    os.makedirs(args.results_dir+"/pvalues")
    
    NUM_QUERIES = args.num_queries
    POSSBILE_SNPS = args.gene_size
    EXIST_COUNT = args.victim_snps
    FIGURES_SAVE_DIR = args.results_dir

    print(args)
    return args

def sharer_u1(ai, y_i, p_prevs, p_currents, num_query, lrt_values):
    q1 = np.quantile(lrt_values, 0.25)
    target_LRTs = lrt_values[lrt_values < q1]

    if len(target_LRTs) != 0:
        target_LRTs = np.append(target_LRTs, np.mean(lrt_values))
        target_LRTs = (target_LRTs - target_LRTs.min()) / \
            (target_LRTs.max() - target_LRTs.min())

    std_dev = np.std(target_LRTs, ddof=1)

    term_one = 1 - std_dev
    if len(target_LRTs) == 0:
        term_one = 1

    return (term_one * 5 + sum(y_i) / len(y_i)) / 6


def attacker_u1(ai, si, p_prev, p_current, num_query):
    utility = p_current

    return utility  # normalize


def retrive_optimal_strategies(victims, num_query, maf_values):
    strategies = np.zeros((victims.shape[1], num_query), dtype=np.int64)

    for ind, victim in enumerate(victims.T):
        strategies[ind] = optimal_scenario(maf_values, num_query, victim)

    return strategies


def retrieve_strategy_for_victim(victim, ind, exist_count, snp_count, possible_snps):
    print(f"Finding strategies for victim {ind}")
    existing_indices = np.where(victim)[0]
    no_existing_indices = np.where(~victim)[0]

    A = np.random.choice(existing_indices, exist_count, replace=False)
    A = np.concatenate((A, np.random.choice(
        no_existing_indices, possible_snps - exist_count, replace=False)))

    while np.sum(victim[A]) < exist_count or not maf_values[A].all() \
            or np.sum(maf_values[A] * victim[A]) >= 0.15 * possible_snps:
        # A = np.random.choice(snp_count, possible_snps)
        A = np.random.choice(existing_indices, exist_count, replace=False)
        A = np.concatenate((A, np.random.choice(
            no_existing_indices, possible_snps - exist_count, replace=False)))

    print(f"Strategies for victim {ind} is: {A}")
    return A


def retrive_random_strategies(exist_count, snp_count, possible_snps):
    strategies = np.zeros((victims.shape[1], possible_snps), dtype=np.int64)

    # Use ThreadPoolExecutor to parallelize
    with ThreadPoolExecutor() as executor:
        futures = []
        for ind, victim in enumerate(victims.T):
            futures.append(executor.submit(retrieve_strategy_for_victim,
                           victim, ind, exist_count, snp_count, possible_snps))

        # Collect the results
        for ind, future in enumerate(futures):
            strategies[ind] = future.result()

    return strategies


# TODO: I think we don't need this func
def retrive_all_attacker(attacker_strategies, num_queries):
    results = []

    for strategy in attacker_strategies:
        results.append(list(permutations(strategy, num_queries)))

    return np.array(results, dtype=np.int64)


def show_strategy_info(strategies, victims, s_beacon):

    for i, strategy in enumerate(strategies):
        A = strategy
        victim = victims[i]

        print("--------------------")
        print(f"User index: {i}")
        print(f"Not exsisting SNP index: {A[~s_beacon[A].any(axis=1)]}")
        print(
            f"Victim: {victim[A]}\nAttacker Strategy Set: {A}\nMAF: {maf_values[A]}")


def default_to_regular(d):
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d


# TODO: very ugly function. fix it.
def func(params):
    a = params[0]  # strategy sets of Attacker
    s = params[1]  # strategy sets of Sharer
    A = params[2]
    S = params[3]
    i = params[4]
    utilities = defaultdict(lambda: defaultdict(lambda: (0, 0)), params[5])

    # utility of sharer
    us = np.array([
        np.array([
            utility_beacon.utility_sharer(np.append(a, ai), np.append(s, si), i)[0][-1] +
            utilities[a+(ai,)][s+(si,)][1] if ai not in a else -np.inf for si in S]) for ai in A
    ])
    # Optimum sharer strategy
    os = np.argmax(us, axis=1)
    bus = np.choose(os, us.T)

    # Find optimum strategy for all possible ai
    ua = np.array([utility_attacker.utility_attacker(np.append(a, ai), np.append(s, S[os[t]]), i, -1)[0][-1] +
                  utilities[a+(ai,)][s+(S[os[t]],)][0] if ai not in a else -np.inf for t, ai in enumerate(A)])
    oa = np.argmax(ua)
    bua = ua[oa]
    return a, s, A[oa], S[os[oa]], bua, bus[oa]


# TODO: very ugly function. fix it.
def build_tree(num_query, attacker_strategy, sharer_strategy):
    utilities = defaultdict(lambda: defaultdict(lambda: (0, 0)))
    strategies = defaultdict(lambda: defaultdict(lambda: (0, 0)))

    i = num_query
    times = []
    while i > 1:
        start = time.time()

        # Generate all possible past strategy combinations
        all_attacker = list(permutations(attacker_strategy, i-1))
        all_sharer = list(product(sharer_strategy, repeat=i-1))
        print(f'i is {i}')

        #  Param grid for multiprocessor pool unit
        paramlist = list(product(all_attacker, all_sharer))
        paramlist = list(map(lambda x: x + (attacker_strategy,
                         sharer_strategy, i, default_to_regular(utilities)), paramlist))
        print('params are generated!')
        pool = multiprocessing.Pool(39)

        try:
            res = pool.map(func, paramlist)
            print('Started all sub processes')
            for r in res:
                strategies[r[0]][r[1]] = (r[2], r[3])
                utilities[r[0]][r[1]] = (r[4], r[5])
            i -= 1

        except:
            print("Something went wrong. Killing all of the processes")
            pool.terminate()
            raise

        finally:
            print('iteration is over. Killing child processes')
            pool.close()
            pool.join()

            end = time.time()
            elapsed_time = end - start
            print(f'Time for {i} is {elapsed_time}')
            times.append(elapsed_time)

    return utilities, strategies


def get_equlibrium_solution(A, S, num_query, utilities, strategies, victim_ind):
    # i = 1
    # Final step, first query
    us = np.array([np.array([utility_beacon.utility_sharer(np.array([ai]), np.array([si]), 1)[
        0][-1] + utilities[tuple([ai])][tuple([si])][1] for si in S]) for ai in A])
    os = np.argmax(us, axis=1)
    bus = np.choose(os, us.T)

    # Find optimum strategy for all possible ai
    ua = np.array([utility_attacker.utility_attacker(np.array([ai]), np.array([S[os[t]]]), 1, victim_ind)
                   [0][-1] + utilities[tuple([ai])][tuple([S[os[t]]])][0] for t, ai in enumerate(A)])
    oa = np.argmax(ua)
    bua = ua[oa]

    #  Finally
    a_eq = np.array([A[oa]])
    s_eq = np.array([S[os[oa]]])
    for i in range(num_query-1):
        a, s = strategies[tuple(a_eq)][tuple(s_eq)]
        a_eq = np.append(a_eq, a, axis=None)
        s_eq = np.append(s_eq, s, axis=None)

    au_eq = np.zeros((num_query+1))
    su_eq = np.zeros((num_query+1))
    for i in range(num_query-1, 0, -1):
        au_eq[i+1], su_eq[i+1] = np.array(utilities[tuple(a_eq[:i])][tuple(
            s_eq[:i])]) - np.array(utilities[tuple(a_eq[:i+1])][tuple(s_eq[:i+1])])
    au_eq[1] = bua - np.array(utilities[tuple(a_eq[:1])][tuple(s_eq[:1])])[0]
    su_eq[1] = bus[oa] - \
        np.array(utilities[tuple(a_eq[:1])][tuple(s_eq[:1])])[1]

    return a_eq, s_eq, au_eq, su_eq


def get_greedy_solution(A, S, num_query, victim_ind):
    a_grd = s_grd = np.array([], dtype=int)
    au_grd = np.zeros((num_query+1))
    su_grd = np.zeros((num_query+1))

    for i in range(num_query):
        # Final step, first query
        us = np.array([np.array([utility_beacon.utility_sharer(np.append(a_grd, ai), np.append(
            s_grd, si), i+1)[0][-1] if ai not in a_grd else -np.inf for si in S]) for ai in A])
        os = np.argmax(us, axis=1)
        bus = np.choose(os, us.T)

        # Find optimum strategy for all possible ai
        ua = np.array([utility_attacker.utility_attacker(np.append(a_grd, ai), np.append(
            s_grd, S[os[t]]), i+1, victim_ind)[0][-1] if ai not in a_grd else -np.inf for t, ai in enumerate(A)])
        oa = np.argmax(ua)
        bua = ua[oa]

        # Obtain best strategies
        a_grd = np.append(a_grd, A[oa])
        s_grd = np.append(s_grd, S[os[oa]])

        # Obtain utilities for the step i
        au_grd[i+1], su_grd[i+1] = (bua, bus[oa])

        # print("A: ", a_grd, "\nS: ", s_grd)
        # print("Utilities: ", au_grd[i+1], su_grd[i+1], "\n")

    return a_grd, s_grd, au_grd, su_grd


def get_optimal_solution(A, num_query, victim_ind):
    a_opt = A[np.argsort(
        maf_values[A][victims.T[victim_ind - 20][A]])][:num_query]
    a_opt = np.concatenate((a_opt, A[np.argsort(
        maf_values[A][1 - victims.T[victim_ind - 20][A]])][:num_query - len(a_opt)]))
    s_opt = np.ones(num_query)

    attacker_utilities = np.zeros(num_query+1)
    sharer_utilities = np.zeros(num_query+1)
    for i in range(num_query):
        attacker_utilities[i+1] = utility_attacker.utility_attacker(
            a_opt[:i+1], s_opt[:i+1], i+1, victim_ind)[0][-1]
        sharer_utilities[i+1] = utility_beacon.utility_sharer(
            a_opt[:i+1], s_opt[:i+1], i+1)[0][-1]

    return a_opt, s_opt, attacker_utilities, sharer_utilities


def plot_beacon_utilities(utilities, solution_name):
    beacon_results = {}

    for i in range(NUM_QUERIES):
        key = f"Query {i+1}"
        beacon_results[key] = utilities[:, i+1]

    fig, ax = plt.subplots()
    # ax.axes.set_yticks(np.arange(0, 1.1, 0.1))
    # ax.set_yticklabels(np.arange(0, 1.1, 0.1))
    ax.set_ylim((0.5, 1.1))
    # ax.set_title(f"{solution_name} - Beacon Utilities")
    ax.boxplot(beacon_results.values())
    ax.set_xticklabels(beacon_results.keys())

    fig.savefig(os.path.join(FIGURES_SAVE_DIR,
                f"{solution_name}_sharer_utility.png"))


def plot_attacker_utilities(utilities, solution_name):
    beacon_results = {}

    for i in range(NUM_QUERIES):
        key = f"Query {i+1}"
        beacon_results[key] = utilities[:, i+1]

    fig, ax = plt.subplots()
    # ax.set_title(f"{solution_name} - Attacker Utilities")
    ax.boxplot(beacon_results.values())
    ax.set_xticklabels(beacon_results.keys())

    fig.savefig(os.path.join(FIGURES_SAVE_DIR,
                f"{solution_name}_attacker_utility.png"))


def box_plot(data, edge_color, fill_color, positions, ax):
    bp = ax.boxplot(data, patch_artist=True, positions=positions, widths=140)

    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)

    return bp


def plot_all_utilities(equlibrium_utilities, greedy_utilities, optimal_utilities, title):
    x = np.array([1, 2, 3, 4]) * 1000
    data1 = np.array(equlibrium_utilities[:, 1:])
    data2 = np.array(greedy_utilities[:, 1:])
    data3 = np.array(optimal_utilities[:, 1:])

    fig, ax = plt.subplots()
    bp1 = box_plot(data1, 'black', 'tomato', x + 175, ax)
    bp2 = box_plot(data2, 'blue', 'cyan', x, ax)
    bp3 = box_plot(data3, 'black', 'orange', x - 175, ax)

    ax.set_xlim(500, 4500)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Query {i+1}' for i in range(len(x))])
    ax.set_ylabel(f"{title} Utility")

    ax.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0]],
              ['Stackelberg', 'Greedy', 'Optimal (no defense)'])
    # ax.set_title(f"Distribution of {title} utilies with 20 victims")

    fig.savefig(os.path.join(FIGURES_SAVE_DIR,
                f"{title}_utilities.png"))


def plot_all_utilities_alertnate(equlibrium_utilities, optimal_utilities, title):
    x = np.array([1, 2, 3, 4]) * 1000
    data1 = np.array(equlibrium_utilities[:, 1:])
    data3 = np.array(optimal_utilities[:, 1:])

    fig, ax = plt.subplots()
    bp1 = box_plot(data1, 'black', 'tomato', x + 175, ax)
    bp3 = box_plot(data3, 'black', 'orange', x - 175, ax)

    ax.set_xlim(500, 4500)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Query {i+1}' for i in range(len(x))])
    ax.set_ylabel(f"{title} Utility")

    ax.legend([bp1["boxes"][0], bp3["boxes"][0]],
              ['Game Theory', 'Optimal (no defense)'])
    # ax.set_title(f"Distribution of {title} utilies with 20 victims")

    fig.savefig(os.path.join(FIGURES_SAVE_DIR,
                f"{title}_alternate_utilities.png"))


args = args_create()
if __name__ == "__main__":
    np.random.seed(args.seed)

    # Extracting column to an array for future use
    maf_values = get_maf_values(args.data_path)
    print("Loaded maf values.")

    maf_values = pre_process_maf(maf_values)
    print("Prepocessed maf values.")

    print("Loading data")
    s_beacon, a_control, victims = get_data(
        args.data_path,
        control_size=args.a_control_size,
        victim_count=args.victim_count,
        beacon_ind_people_size=args.beacon_size,
        victims_in_beacon_count=args.victims_in_beacon_count
    )

    print("Loaded and split data (Victims, beacon, attacker control group)")
    print(f'Beacon size: {len(s_beacon.T)}')
    print(f'Attacker control group size: {len(a_control.T)}')
    print(f'Victims in beacon: {args.victims_in_beacon_count}')

    randomness_id = np.random.rand()
    print(f'Randomness Id is: {randomness_id}')

    # Utility Functions
    afunc = attacker_u1
    sfunc = sharer_u1

    # Utility helper classes
    utility_attacker = UtilityAttacker(maf_values, a_control, afunc)
    utility_beacon = UtilityBeacon(maf_values, s_beacon, sfunc)

    # Sharer's Strategy
    sharer_strategies = np.array([0.5, 0.75, 1, 0.25])

    print("Calculating attacker's strategy.")
    all_sharer = list(product(sharer_strategies, repeat=NUM_QUERIES))

    # Calculate attackers' strategy
    attackers_strategies = retrive_random_strategies(
        EXIST_COUNT, len(maf_values), POSSBILE_SNPS)
    all_attacker = retrive_all_attacker(attackers_strategies, NUM_QUERIES)

    total_iteration_count = all_attacker.shape[0] * \
        all_attacker.shape[1] * len(all_sharer)
    print("\n*** Strategy info - Start ***")
    print(f"Beacon Strategy Set: {sharer_strategies}")
    print(f"Will try at most {total_iteration_count} iterations")

    show_strategy_info(attackers_strategies, victims.T, s_beacon)

    print("*** Strategy info - End ***\n")
    optimal_sharer_strategies = np.zeros((victims.shape[1], NUM_QUERIES))
    optimal_attacker_strategies = np.zeros(
        (victims.shape[1], NUM_QUERIES), dtype=np.int64)
    optimal_sharer_utilities = np.zeros((victims.shape[1], NUM_QUERIES+1))
    optimal_attacker_utilities = np.zeros((victims.shape[1], NUM_QUERIES+1))
    print("\n*** Optimal Solution - Start ***")
    for i in range(victims.shape[1]):
        print("--------------------")
        print(f"User index: {i}")
        optimal_attacker_strategies[i], optimal_sharer_strategies[i], optimal_attacker_utilities[i],  optimal_sharer_utilities[i] = get_optimal_solution(
            attackers_strategies[i], NUM_QUERIES, 20+i)
        print(
            f"Strategies # Attacker: {optimal_attacker_strategies[i]}, Beacon: {optimal_sharer_strategies[i]}")
        print(
            f"Utilities # Attacker: {optimal_attacker_utilities[i]}, Beacon: {optimal_sharer_utilities[i]}")


    print("*** Building Tree - Start ***")

    strategies = []
    utilities = []
    for i in range(victims.shape[1]):
        print("--------------------")
        print(f"User index: {i}")
        utility, strategy = build_tree(
            NUM_QUERIES, attackers_strategies[i], sharer_strategies)

        strategies.append(strategy)
        utilities.append(utility)

    print("*** Building Tree - End ***\n")

    equlibrium_sharer_strategies = np.zeros((victims.shape[1], NUM_QUERIES))
    equlibrium_attacker_strategies = np.zeros(
        (victims.shape[1], NUM_QUERIES), dtype=np.int64)
    equlibrium_sharer_utilities = np.zeros((victims.shape[1], NUM_QUERIES+1))
    equlibrium_attacker_utilities = np.zeros((victims.shape[1], NUM_QUERIES+1))
    print("\n*** Equlibrium Solution - Start ***")
    for i in range(victims.shape[1]):
        print("--------------------")
        print(f"User index: {i}")
        equlibrium_attacker_strategies[i], equlibrium_sharer_strategies[i], equlibrium_attacker_utilities[i],  equlibrium_sharer_utilities[i] = get_equlibrium_solution(
            attackers_strategies[i], sharer_strategies, NUM_QUERIES, utilities[i], strategies[i], 20 + i)
        print(
            f"Strategies # Attacker: {equlibrium_attacker_strategies[i]}, Beacon: {equlibrium_sharer_strategies[i]}")
        print(
            f"Utilities # Attacker: {equlibrium_attacker_utilities[i]}, Beacon: {equlibrium_sharer_utilities[i]}")

    print("*** Equlibrium Solution - End ***\n")

    print("Generating plots...")
    plot_beacon_utilities(equlibrium_sharer_utilities, "Equlibrium")
    plot_attacker_utilities(equlibrium_attacker_utilities, "Equlibrium")
    print("Plots are generated.")

    greedy_sharer_strategies = np.zeros((victims.shape[1], NUM_QUERIES))
    greedy_attacker_strategies = np.zeros(
        (victims.shape[1], NUM_QUERIES), dtype=np.int64)
    greedy_sharer_utilities = np.zeros((victims.shape[1], NUM_QUERIES+1))
    greedy_attacker_utilities = np.zeros((victims.shape[1], NUM_QUERIES+1))
    print("\n*** Greedy Solution - Start ***")
    for i in range(victims.shape[1]):
        print("--------------------")
        print(f"User index: {i}")
        greedy_attacker_strategies[i], greedy_sharer_strategies[i], greedy_attacker_utilities[i],  greedy_sharer_utilities[i] = get_greedy_solution(
            attackers_strategies[i], sharer_strategies, NUM_QUERIES, 20 + i)
        print(
            f"Strategies # Attacker: {greedy_attacker_strategies[i]}, Beacon: {greedy_sharer_strategies[i]}")
        print(
            f"Utilities # Attacker: {greedy_attacker_utilities[i]}, Beacon: {greedy_sharer_utilities[i]}")

    print("*** Greedy Solution - End ***\n")

    print("Generating plots...")
    plot_beacon_utilities(greedy_sharer_utilities, "Greedy")
    plot_attacker_utilities(greedy_attacker_utilities, "Greedy")
    print("Plots are generated.")


    print("*** Optimal Attack - End ***\n")

    # Save the data
    print("Saving data...")
    with open(os.path.join(FIGURES_SAVE_DIR,
                        f"plot_data.npy"), 'wb') as f:
        np.save(f, equlibrium_attacker_utilities)
        np.save(f, greedy_attacker_utilities)
        np.save(f, optimal_attacker_utilities)
        np.save(f, equlibrium_sharer_utilities)
        np.save(f, greedy_sharer_utilities)
        np.save(f, optimal_sharer_utilities)

    print("Generating plots...")
    plot_beacon_utilities(optimal_sharer_utilities, "Optimal Attack")
    plot_attacker_utilities(optimal_attacker_utilities, "Optimal Attack")
    print("Plots are generated.")


    print("Generating overall plots...")
    plot_all_utilities(equlibrium_attacker_utilities,
                       greedy_attacker_utilities, optimal_attacker_utilities, "Attacker")
    plot_all_utilities(equlibrium_sharer_utilities,
                       greedy_sharer_utilities, optimal_sharer_utilities, "Beacon")

    plot_all_utilities_alertnate(
        equlibrium_attacker_utilities, optimal_attacker_utilities, "Attacker")
    plot_all_utilities_alertnate(
        equlibrium_sharer_utilities, optimal_sharer_utilities, "Beacon")

    print("Plots are generated.")
