from collections import defaultdict
from itertools import permutations, product
import multiprocessing
import time
import numpy as np
import warnings
import matplotlib.pyplot as plt

from common.data import get_maf_values, pre_process_maf, get_data
from common.lrt import get_optimal_lrt_victim, get_optimal_lrt, p_value_ind, calculate_lrt
from common.scenario import optimal_scenario
from common.utility import UtilityAttacker, UtilityBeacon

warnings.filterwarnings('ignore')

# TODO: load these from cmd params
NUM_QUERIES = 4
POSSBILE_SNPS = 4
EXIST_COUNT = 4
BEACON_NOT_EXIST_COUNT = 0


def sharer_u1(ai, si, p_prevs, p_currents, num_query, lrt_values):
    q1 = np.quantile(lrt_values, 0.25)
    target_LRTs = lrt_values[lrt_values < q1]

    if len(target_LRTs) != 0:
        target_LRTs = np.append(target_LRTs, np.mean(lrt_values))
        target_LRTs = (target_LRTs - target_LRTs.min()) / \
            (target_LRTs.max() - target_LRTs.min())

    mean = np.mean(target_LRTs)
    std_dev = np.std(target_LRTs, ddof=1)

    term_one = (1 - (std_dev / mean)
                ) if mean != 0 and len(target_LRTs) != 0 else 1

    return (term_one * 5 + sum(si) / len(si)) / 6


def attacker_u1(ai, si, p_prev, p_current, num_query):
    utility = (p_current - p_prev)
    # print(f'* Attacker: ai: {ai}, si: {si}, p_prev: {p_prev}, p_current: {p_current}, num_query: {num_query} \n utility: {(utility+2) * 2/3}')
    return utility  # normalize
    # return (utility+2) * 2/3 #normalize


def retrive_optimal_strategies(victims, num_query, maf_values):
    strategies = np.zeros((victims.shape[1], num_query), dtype=np.int64)

    for ind, victim in enumerate(victims.T):
        strategies[ind] = optimal_scenario(maf_values, num_query, victim)

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


if __name__ == "__main__":
    np.random.seed(2024)

    mainPath = "./data"

    # Extracting column to an array for future use
    maf_values = get_maf_values(mainPath)
    print("Loaded maf values.")

    maf_values = pre_process_maf(maf_values)
    print("Prepocessed maf values.")

    print("Loading data")
    s_beacon, a_control, victims = get_data(
        mainPath, victims_in_beacon_count=20)

    # TODO: check the existance of victims in both beacon and control group

    print("Loaded and split data (Victims, beacon, attacker control group)")
    print(f'Beacon size: {len(s_beacon.T)}')
    print(f'Attacker control group size: {len(a_control.T)}')
    print(f'Victims in beacon: {10}')

    randomness_id = np.random.rand()
    print(f'Randomness Id is: {randomness_id}')

    # Utility Functions
    afunc = attacker_u1
    sfunc = sharer_u1

    # Utility helper classes
    utility_attacker = UtilityAttacker(maf_values, a_control, afunc)
    utility_beacon = UtilityBeacon(maf_values, s_beacon, sfunc)

    # Sharer's Strategy
    sharer_strategies = np.array([0.5, 0.7, 1, 0.25])

    print("Print calculating attacker's strategy.")
    all_sharer = list(product(sharer_strategies, repeat=NUM_QUERIES))

    # Calculate attackers' strategy
    attackers_strategies = retrive_optimal_strategies(
        victims, NUM_QUERIES, maf_values)
    all_attacker = retrive_all_attacker(attackers_strategies, NUM_QUERIES)

    total_iteration_count = all_attacker.shape[0] * \
        all_attacker.shape[1] * len(all_sharer)
    print("\n*** Strategy info - Start ***")
    print(f"Beacon Strategy Set: {sharer_strategies}")
    print(f"Will try at most {total_iteration_count} iterations")

    show_strategy_info(attackers_strategies, victims.T, s_beacon)

    print("*** Strategy info - End ***")

    print("\n")
    print("Building Tree")

    strategies = []
    utilities = []
    for i in range(victims.shape[1]):
        print("--------------------")
        print(f"User index: {i}")
        utility, strategy = build_tree(
            NUM_QUERIES, attackers_strategies[i], sharer_strategies)

        strategies.append(strategy)
        utilities.append(utility)
