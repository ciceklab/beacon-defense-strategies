import numpy as np


# This function is very ugly. I agree :(
def random_scenario(victim, snp_count, possible_snps, maf_values, exist_count, beacon_not_exist_count, num_query, s_beacon, a_control):
    A = np.random.choice(snp_count, possible_snps)

    while np.sum(victim[A]) < exist_count or not maf_values[A].all() \
            or np.sum(maf_values[A] * victim[A]) >= 0.15 * exist_count:
        # or (num_query - s_beacon[A].any(axis=1).sum() < beacon_not_exist_count) \
        # or num_query - a_control[A].any(axis=1).sum() > beacon_not_exist_count:
        # print(np.sum(victim[A]))
        A = np.random.choice(snp_count, possible_snps)

    return A


def optimal_scenario(maf_values, num_query, victim):
    # Select rarest of victim
    sorted_indx = np.argsort(maf_values)
    # mask = np.isin(sorted_indx, np.where((victim == 1) & (s_beacon.sum(axis=1) - victim > 0))[0])
    mask = np.isin(sorted_indx, np.where((victim == 1))[0])
    snps = sorted_indx[mask][100:100 + num_query]

    return snps
