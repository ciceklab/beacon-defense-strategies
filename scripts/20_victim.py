from itertools import permutations, product
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