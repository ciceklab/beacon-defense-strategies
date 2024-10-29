import pickle
from os.path import join

import numpy as np
import pandas as pd


def get_maf_values(mainPath):
    # Table that contains MAF (minor allele frequency) values for each position.
    maf = pd.read_csv(join(mainPath, "MAF.txt"),
                      index_col=0, delim_whitespace=True)
    maf.rename(columns={'referenceAllele': 'major', 'referenceAlleleFrequency': 'major_freq',
                        'otherAllele': 'minor', 'otherAlleleFrequency': 'minor_freq'}, inplace=True)
    maf["maf"] = np.round(maf["maf"].values, 3)

    # Extracting column to an array for future use
    maf_values = maf["maf"].values

    return maf_values


def pre_process_maf(maf_values):
    return np.array([0.001 if x == 0 else x for x in maf_values])


def get_data(mainPath, control_size=40, victim_count=20, beacon_ind_people_size=60, victims_in_beacon_count=10):
    if control_size < victim_count:
        raise ValueError(
            "victims are included in the control group so vitim count should be lower than or equal to control group size")

    if (victims_in_beacon_count > victim_count):
        raise ValueError(
            "The count of victims in the beacon should be at most equal to count of vicitms")

    #  CEU Beacon - it contains 164 people in total which we will divide into groups to experiment
    beacon = pd.read_csv(join(mainPath, "Beacon_164.txt"),
                         index_col=0, delim_whitespace=True)

    # Reference genome, i.e. the genome that has no SNPs, all major allele pairs for each position
    reference = pickle.load(open(join(mainPath, "reference.pickle"), "rb"))

    # Binary representation of the beacon; 0: no SNP (i.e. no mutation) 1: SNP (i.e. mutation)
    binary = np.logical_and(beacon.values != reference,
                            beacon.values != "NN").astype(int)

    # Prepare index arrays for future use
    # beacon_people = np.arange(65)
    # other_people = np.arange(99)+65
    all_people = np.arange(164)

    shuffled = np.random.permutation(all_people)

    victim_ind = shuffled[control_size - victim_count:control_size]
    attacker_control_ind = shuffled[:control_size]
    beacon_ind = shuffled[control_size:control_size+beacon_ind_people_size]

    s_beacon = binary[:, np.concatenate([beacon_ind, np.array(
        victim_ind[:victims_in_beacon_count])])]  # Victim inside beacon

    a_control = binary[:, attacker_control_ind]

    victims = binary[:, victim_ind]

    return s_beacon, a_control, victims
