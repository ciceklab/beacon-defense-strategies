import numpy as np
from common.lrt import get_optimal_lrt, get_optimal_lrt_victim, p_value, p_value_ind


class UtilityAttacker:
    def __init__(self, maf_values, control_group, attacker_utility_fn) -> None:
        self.maf_values = maf_values
        self.attacker_utility_fn = attacker_utility_fn
        self.control_group = control_group

    def __helper(self, ai, si, victim_ind, p_prev_victim) -> tuple:
        query_length = len(si)

        # Current p-value
        victim_delta, control_delta = get_optimal_lrt_victim(
            victim_ind, self.maf_values, self.control_group, ai, si)
        p_victim_current = 1 - p_value(victim_delta, control_delta)
        utility = self.attacker_utility_fn(
            ai[-1], si[-1], p_prev_victim, p_victim_current, query_length)

        return utility, p_victim_current

    def utility_attacker(self, a, s, num_query, victim_ind):
        # Attacker Utility
        utility = np.zeros(num_query)

        # Previous p-value
        p_victim_prev = 0
        p_values = np.zeros(num_query)
        for i in range(num_query):
            utility[i], p_victim_prev = self.__helper(
                a[:i+1], s[:i+1], victim_ind, p_victim_prev)
            p_values[i] = p_victim_prev

        return utility, p_values


class UtilityBeacon:
    def __init__(self, maf_values, beacon, beacon_utility_fn) -> None:
        self.maf_values = maf_values
        self.beacon = beacon
        self.beacon_utility_fn = beacon_utility_fn

    def __helper(self, ai, si, p_prev_donors) -> tuple:
        query_length = len(si)

        p_donors_current = np.zeros(self.beacon.shape[1])
        lrt_values = get_optimal_lrt(self.maf_values, self.beacon, ai, si)

        for j in range(self.beacon.shape[1]):
            p_donors_current[j] = p_value_ind(lrt_values, j)

        utility = self.beacon_utility_fn(
            ai[-1], si, p_prev_donors, p_donors_current, query_length, lrt_values)

        return utility, p_donors_current, lrt_values

    def utility_sharer(self, a, s, num_query):
        # Sharer Utility
        utility = np.zeros(num_query)
        lrt_values = np.zeros((num_query, self.beacon.shape[1]))

        # Previous p-value
        p_donors_prev = np.ones(self.beacon.shape[1])
        for i in range(num_query):
            utility[i], p_donors_prev, lrt_values[i] = self.__helper(
                a[:i+1], s[:i+1], p_donors_prev)

        return utility, lrt_values
