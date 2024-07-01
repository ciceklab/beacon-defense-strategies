import numpy as np


def calculate_A(maf, num_people, error=0.001):
    if (len(maf) == 0):
        return []

    DN_i = np.power((1-maf), (2*num_people))
    DN_i_1 = np.power((1-maf), (2*num_people-2))

    return np.log(1 - DN_i) - np.log(1 - error * DN_i_1)


def calculate_B(maf, num_people, error=0.001):
    if (len(maf) == 0):
        return []

    DN_i = np.power((1-maf), (2*num_people))
    DN_i_1 = np.power((1-maf), (2*num_people-2))

    return np.log(DN_i) - np.log(error*DN_i_1)


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


def get_optimal_lrt(maf_values, control_people, A, S):
    num_query = len(A)
    control_size = control_people.shape[1]

    # Victim lrt
    response = control_people[A].any(axis=1)
    maf_i = maf_values[A]

    control_lrt = np.zeros((control_size, num_query))
    for i in range(control_size):
        maf_i = maf_values[A]
        control_lrt[i] = calculate_lrt(
            control_size,
            S,
            control_people[A, i],
            maf_i,
            response
        )

    return np.sum(control_lrt, axis=1)


def get_optimal_lrt_victim(victim_ind, maf_values, control_people, A, S):
    control_size = control_people.shape[1]
    control_lrt = get_optimal_lrt(maf_values, control_people, A, S)

    mask = np.ones((control_size), dtype=np.bool_())
    mask[victim_ind] = False

    return control_lrt[victim_ind], control_lrt[mask]


def p_value(victim_lrt, control_lrt):
    return np.sum(control_lrt <= victim_lrt) / control_lrt.shape[0]


def p_value_ind(control_lrts, victim_ind):
    mask = np.ones_like(control_lrts, dtype=np.bool_())
    mask[victim_ind] = False

    return p_value(control_lrts[victim_ind], control_lrts[mask])
