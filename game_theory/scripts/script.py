# %%
import os
from IPython.core.display import display, HTML
import pandas as pd
import numpy as np
import random
import math
import itertools
import warnings
import pickle
import gc
import sys
import matplotlib.pyplot as plt
from os.path import join, exists
from collections import Counter, defaultdict
from scipy.special import gamma
from itertools import permutations, combinations, combinations_with_replacement, product
import timeit
import multiprocessing
import tqdm
import time
warnings.filterwarnings('ignore')
np.set_printoptions(suppress=True, formatter={
                    'float': lambda x: "{0:0.5f}".format(x)})
# display(HTML("<style>.container { width:75% !important; }</style>"))

# %% [markdown]
# ### Step 1: Load Beacon, MAF, Reference and other cached variables
print("Starting")
# %%
# Replace it with the main folder
mainPath = "./data"
np.random.seed(2024)

# %%
#  CEU Beacon - it contains 164 people in total which we will divide into groups to experiment
beacon = pd.read_csv(join(mainPath, "Beacon_164.txt"),
                     index_col=0, delim_whitespace=True)

# Reference genome, i.e. the genome that has no SNPs, all major allele pairs for each position
reference = pickle.load(open(join(mainPath, "reference.pickle"), "rb"))

# Binary representation of the beacon; 0: no SNP (i.e. no mutation) 1: SNP (i.e. mutation)
binary = np.logical_and(beacon.values != reference,
                        beacon.values != "NN").astype(int)

# Table that contains MAF (minor allele frequency) values for each position.
maf = pd.read_csv(join(mainPath, "MAF.txt"),
                  index_col=0, delim_whitespace=True)
maf.rename(columns={'referenceAllele': 'major', 'referenceAlleleFrequency': 'major_freq',
                    'otherAllele': 'minor', 'otherAlleleFrequency': 'minor_freq'}, inplace=True)
maf["maf"] = np.round(maf["maf"].values, 3)

print("loaded MAFs")

# Same variable with sorted maf values
sorted_maf = maf.sort_values(by='maf')

# Extracting column to an array for future use
maf_values = maf["maf"].values

# Prepare index arrays for future use
# beacon_people = np.arange(65)
# other_people = np.arange(99)+65
all_people = np.arange(164)

# %%
# MAF values are calculated from a small subset they might be 0.
# This does not mean they are not seen in anybody in the world so we are replacing 0 MAF values with 0.001 which is a pretty rare value
maf["maf"] = maf["maf"].apply(lambda x: 0.001 if x == 0 else x)
maf_values = maf["maf"].values

# %% [markdown]
# #### Build beacons, control groups and select victim
# Construct beacons and the victim. Here randomly select;
# 1. A single victim
# 2. 20 control group people for attacker to calculate LRT values
# 3. 20 control group people for sharer to calculate LRT values
# 4. 60 people beacon for sharer

# %%
shuffled = np.random.permutation(all_people)

victim_ind = shuffled[0]
a_cind = shuffled[1:21]
s_cind = shuffled[41:61]

s_ind = shuffled[81:140]

s_beacon = binary[:, np.concatenate(
    [s_ind, np.array([victim_ind])])]  # Victim inside beacon

a_control = binary[:, a_cind]
s_control = binary[:, s_cind]

victim = binary[:, victim_ind]

randomness_id = np.random.rand()

print("Done splitting data.")

# %% [markdown]
# ### Step 2: Function definitions

# %% [markdown]
# #### Position based LRT calculation
# First function calculates LRT value for given number of people in beacon, it's response, victim's genome and maf values
#
# Second one calculates LRT valus for a single person and a given control group, then returns it

# %%


def calculate_A(maf, num_people):
    if (len(maf) == 0):
        return []

    DN_i = np.power((1-maf), (2*num_people))
    DN_i_1 = np.power((1-maf), (2*num_people-2))

    return np.log(1 - DN_i) - np.log(1 - error * DN_i_1)


def calculate_B(maf, num_people):
    if (len(maf) == 0):
        return []

    DN_i = np.power((1-maf), (2*num_people))
    DN_i_1 = np.power((1-maf), (2*num_people-2))

    return np.log(DN_i) - np.log(error*DN_i_1)


# %%
# Masoud Version
# n: Num query
error = 0.001


def calculate_lrt(num_people, y, genome, maf, actual_responses):
    # index of queries which has response zero
    Q_0 = np.where(~actual_responses)
    # index of queroes which has  response one
    Q_1 = np.where(actual_responses)

    # Impact of beacon's zero answers
    zero_flip = np.sum(genome[Q_0] * calculate_B(maf[Q_0], num_people))

    y = y[Q_1]
    # Impact of beacon's one answers
    one_flip = np.sum(
        genome[Q_1] * (
            y * calculate_A(maf[Q_1], num_people) +
            (1 - y) * calculate_B(maf[Q_1], num_people)
        )
    )
    # print(f"Strategy: {S}")
    # print(f"Actual response: {actual_responses}")
    # print(f"Is SNP: {genome}")
    # print(f"SNP impact: {SNP_impact}, non SNP impact: {non_SNP_impact}, LRT: {SNP_impact + non_SNP_impact}")
    # print("====================================")

    LRT = zero_flip + one_flip
    return LRT


def optimal_lrt(victim, control_people, control_targets, beacon, A, S, num_query):
    control_size = control_people.shape[1]
    beacon_size = beacon.shape[1]

    # Victim lrt
    response = beacon[A].any(axis=1)
    # coef = (2*S-1)*response + 1 - S
    maf_i = maf_values[A]
    # print("**************")
    # print("*** Victim ***")
    # print(f"victim: {victim}")
    # print(f"A: {A}")
    # print(f"victim[A]: {victim[A]}")
    victim_lrt = calculate_lrt(beacon_size, S, victim[A], maf_i, response)

    # Find the SNPs to query for control lrt calculation (pre-computed before)
    # Bottelneck
    # control_asked = np.array([
    #     control_targets[np.searchsorted(
    #         np.unique(victim_snps[victim[a]]),
    #         maf_values[a]
    #     )][:, victim[a]] for a in A]).T

    control_asked = [A.copy() for _ in range(control_size)]

    #  Query the beaconS
    responses = np.array([beacon[target].any(axis=1)
                         for target in control_asked])
    control_lrt = np.zeros((control_size, num_query))
    for i in range(control_size):
        target = control_asked[i]
        maf_i = maf_values[target]
        # control_lrt[i] = calculate_lrt(beacon_size, np.ones(S.shape), control_people[target, i], maf_i, responses[i])  # we may have problem here!
        control_lrt[i] = calculate_lrt(
            beacon_size, S, control_people[target, i], maf_i, responses[i])  # we may have problem here!

    return np.sum(victim_lrt), np.sum(control_lrt, axis=1)

# %% [markdown]
# ###### p-value Function

# %%


def p_value(victim_lrt, control_lrt):
    return np.sum(control_lrt <= victim_lrt) / control_lrt.shape[0]

# %% [markdown]
# ##### Helpers calculate utility for specific step i and returns it along with the previous p_values

# %%


def ua_helper(utility_func, ai, si, i, p_prev_victim):
    # Current p-value
    victim_delta, control_delta = optimal_lrt(
        victim, a_control, None, s_beacon, ai, si, i)
    p_victim_current = p_value(victim_delta, control_delta)
    utility = utility_func(ai[-1], si[-1], p_prev_victim, p_victim_current, i)
    return utility, p_victim_current


def us_helper(utility_func, ai, si, i, p_prev_donors):
    # print(f"si: {si}")
    # Current p-value
    p_donors_current = np.zeros(s_beacon.shape[1])
    for j in range(s_beacon.shape[1]):
        # victim_delta, control_delta = optimal_lrt(s_beacon[:, j], s_control, s_control_targets, s_beacon, ai, si, i)
        victim_delta, control_delta = optimal_lrt(
            s_beacon[:, j], s_beacon, None, s_beacon, ai, si, i)
        p_donors_current[j] = p_value(victim_delta, control_delta)

    # print(f"strategy is: {si}")
    # print(f"Current p values: {p_donors_current}")
    utility = utility_func(ai[-1], si, p_prev_donors, p_donors_current, i)
    return utility, p_donors_current

# %% [markdown]
# ##### Utility functions calculate utility value for the LAST step

# %%


def utility_attacker(a, s, num_query, utility_func):
    # Attacker Utility
    utility = np.zeros(num_query)
    # Previous p-value
    p_victim_prev = 1
    for i in range(num_query):
        utility[i], p_victim_prev = ua_helper(
            utility_func, a[:i+1], s[:i+1], i+1, p_victim_prev)
    return utility[-1]


def utility_sharer(a, s, num_query, utility_func):
    # Sharer Utility
    utility = np.zeros(num_query)
    # Previous p-value
    p_donors_prev = np.ones(s_beacon.shape[1])
    for i in range(num_query):
        utility[i], p_donors_prev = us_helper(
            utility_func, a[:i+1], s[:i+1], i+1, p_donors_prev)
    return utility[-1]

# %% [markdown]
# ## Utility Functions

# %% [markdown]
# **MAF Value:** <code>-np.log(maf.iloc[ai]["maf"]+1e-3)/abs(np.log(1e-3))</code> <br>
# **Lying Probability:** <code>(1-si)</code><br>
# **Number of Queries Sent:** <code>num_query/20</code><br>
# **p-value change for donors:** <code>np.sum(p_prevs - p_currents)/len(p_prevs) </code><br>
# **Count of risky people:** <code>np.sum(p_currents <= 0.05)/len(p_currents)</code><br>
# **Deviation of p-values:** <code>np.std(p_currents)</code><br>

# %%


def sharer_u1(ai, si, p_prevs, p_currents, num_query):
    # utility = 1 * np.sum(1-si) + np.std(p_currents)*3
    # utility = np.std(p_currents) * 2 + len(si) / sum(1-si) + 2*(1-si[-1])
    # print(f'* Sharer - ai: {ai}, si: {si}, p_prevs: {p_prevs}, p_currents: {p_currents}, num_query: {num_query} \n utility: {(-utility+6)/(2)}, new term1: { len(si) / sum(1-si)}, {2*(1-si[-1])}')
    # return (-utility+2)/(2) #normalize
    mean = np.mean(p_currents)
    # ddof=1 for sample standard deviation
    std_dev = np.std(p_currents, ddof=1)

    # utility = (
    #         - (std_dev / mean) * 2 +
    #         sum(si) / len(si)
    #     ) / 2

    # First term with a weight of 2 for a privacy hone
    # Second term with a weight of 1 for a utilty hone
    utility = (1 - (std_dev / mean)) * 3 + sum(si) / len(si)

    return utility / 4

# %%


def attacker_u1(ai, si, p_prev, p_current, num_query):
    utility = (p_prev - p_current)
    # print(f'* Attacker: ai: {ai}, si: {si}, p_prev: {p_prev}, p_current: {p_current}, num_query: {num_query} \n utility: {(utility+2) * 2/3}')
    return utility  # normalize
    # return (utility+2) * 2/3 #normalize


# %%
# Utility Functions
afunc = attacker_u1
sfunc = sharer_u1

# %% [markdown]
# ### Game Scenario

# %%
num_query = 6
possible_snps = 6
exist_count = 4
beacon_not_exist_count = 0

if (exist_count + beacon_not_exist_count > num_query):
    raise ValueError("check the values")

# %%
# Find rare SNPs
A = np.random.choice(s_beacon.shape[0], possible_snps)
# 1- ensure that
# 2- ensure that the selected SNPs are reliable. (They have a good MAF)
# 3- ensure that some SNPs, given their counts, do not exist at the beacon.
# 4- ensure that the SNPs not existing at the beacon exists at the attacker's control.
while np.sum(victim[A]) < exist_count or not maf.iloc[A]["maf"].values.all() \
        or np.sum(maf.iloc[A]["maf"]*victim[A]) >= 0.15*exist_count \
        or (num_query - s_beacon[A].any(axis=1).sum() < beacon_not_exist_count) \
        or num_query - a_control[A].any(axis=1).sum() > beacon_not_exist_count:
    # print(np.sum(victim[A]))
    A = np.random.choice(s_beacon.shape[0], possible_snps)

# %%
s_beacon[2406982]

# %%
# Select rarest of victim
# A = sorted_maf.loc[sorted_maf.index.isin(np.where(victim)[0])]["maf"]
# A = A.index[100:100+num_query].values

# %%
#  Sharer's Strategy
S = np.round(np.arange(0.7, 1.01, 0.1), 2)
S = np.array([0.5, 0.7, 1])

# Product Space
all_attacker = np.array(list(permutations(A, num_query)))
all_sharer = list(product(S, repeat=num_query))


# Utility Functions
afunc = attacker_u1
sfunc = sharer_u1

# Print Information
print("Will try at most ", len(all_attacker)*len(all_sharer), " iterations")
print(f"Not exsisting SNP index: {A[~s_beacon[A].any(axis=1)]}")
print("Victim: ", victim[A], "\nAttacker Strategy Set: ", A,
      "\nSharer Strategy Set: ", S, "\nMAF: ", maf.iloc[A]["maf"].values)

# %%

calculate_lrt(60, np.array([1]), np.array([1]), np.array(
    [maf.iloc[878787]["maf"]]), np.array([False]))

# %%
maf.iloc[2309239]["maf"]

# %% [markdown]
# ## 1. Equilibrium

# %% [markdown]
# #### Disclaimer from kayoz@
# This looks ugly and I wrote it long time ago. It is working perfectly fine but feel free to make it faster and more beautiful
#
# Basically it traverses the whole game tree bottom-up to calculate which path leads to Nash Equilibrium
#
# It's also multithreaded (as much as it can be in Python) so I suggest not to touch in below code if not necessary or if you are not crazy ambitious


# %%

utilities = defaultdict(lambda: defaultdict(lambda: (0, 0)))
strategies = defaultdict(lambda: defaultdict(lambda: (0, 0)))


def func(params):
    a = params[0]  # strategy sets of Attacker
    s = params[1]  # strategy sets of Sharer
    # print(f"{a}, {s}")
    # utility of sharer
    us = np.array([
        np.array([
            utility_sharer(np.append(a, ai), np.append(s, si), i, sfunc) + utilities[a+(ai,)][s+(si,)][1] if ai not in a else -np.inf for si in S]) for ai in A
    ])
    # Optimum sharer strategy
    os = np.argmax(us, axis=1)
    bus = np.choose(os, us.T)

    # Find optimum strategy for all possible ai
    ua = np.array([utility_attacker(np.append(a, ai), np.append(s, S[os[t]]), i, afunc) +
                  utilities[a+(ai,)][s+(S[os[t]],)][0] if ai not in a else -np.inf for t, ai in enumerate(A)])
    oa = np.argmax(ua)
    bua = ua[oa]
    return a, s, A[oa], S[os[oa]], bua, bus[oa]


i = num_query
times = []
while i > 1:
    start = time.time()

    # Generate all possible past strategy combinations
    all_attacker = list(permutations(A, i-1))
    all_sharer = list(product(S, repeat=i-1))
    print(f'i is {i}')

    #  Param grid for multiprocessor pool unit
    paramlist = list(product(all_attacker, all_sharer))
    print('params are generated!')
    pool = multiprocessing.Pool(50)

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
        break

    finally:
        print('iteration is over. Killing child processes')
        pool.close()
        pool.join()

        end = time.time()
        elapsed_time = end - start
        print(f'Time for {i} is {elapsed_time}')
        times.append(elapsed_time)

    # print(f'Running with params {paramlist[0]}')
    # test_x = func(paramlist[0])
    # break

with open('test.npy', 'wb') as f:
    np.save(f, np.array(times))

# %%
np.load("test.npy")

# %%
plt.xlabel("Query length")
plt.ylabel("Time elapsed (seconds)")
plt.xticks(np.arange(0, 6))
plt.plot(np.arange(2, 6), np.round(np.array(times[::-1])))
plt.savefig("output.png")
