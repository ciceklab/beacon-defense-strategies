# A Reinforcement Learning-based Approach for Dynamic Privacy Protection in Genomic Data Sharing Beacons

> This is the first reinforcement learning (RL)-based approach to dynamically defend the beacon protocol against threats. Our multi-agent RL framework trains two agents: (i) a "Generic-Beacon-Defender" (GBD) that adaptively adjusts response rates to optimize privacy and utility, and (ii) a "Generic-Beacon-Attacker" (GBA) that intelligently selects query orders and submits random queries to mimic regular user behavior. This novel, adaptive defense mechanism responds in real time to distinguish between legitimate users and potential attackers, applying targeted strategies for enhanced protection.

> **Keywords**: <a href="https://en.wikipedia.org/wiki/Reinforcement_learning" target="_blank">**Reinforcement Learning**</a>, <a href="https://arxiv.org/abs/1802.09477" target="_blank">**TD3**</a>, <a href="https://spinningup.openai.com/en/latest/algorithms/ppo.html" target="_blank">**PPO**</a>, <a href="https://en.wikipedia.org/wiki/Game_theory" target="_blank">**Game Theory**</a>.

---

## Authors

Masoud Poorghaffar Aghdam, Sobhan Shukueian Tabrizi, Kerem Ayöz, Erman Ayday, Sinem Sav, and A. Ercüment Çiçek


---

## Table of Contents 

> **Note**: This framework is open for academic use but requires licensing for commercial use. Please refer to the [License](#license) section for more details.

- [Installation](#installation)
- [Features](#features)
- [Instructions Manual](#instructions-manual)
- [Citations](#citations)
- [License](#license)

---

## Installation

The Beacon RL Framework requires the dependencies listed in `environment.yml`. 

### Requirements

To set up the project environment, run the following steps:

1. Install Conda
2. Create and activate the Conda environment using `environment.yml` as follows:

```shell
conda env create --name beacon -f environment.yml
conda activate beacon
```
---

## Instructions Manual

### Training the Models

> **Important**: Run the training scripts from the main directory.

The following command trains both attacker and beacon models. Specify options like agent type and update frequency.

```shell
python train.py --train "both" --episodes 100000 --update_freq 10 --max_queries 100
```

#### Key Arguments

##### Training Setup
-  `--episodes`: Sets the total number of episodes for training.
-	`--train`: Specifies which agent(s) to train: attacker, beacon, or both.
-	`--attacker_type`: Type of attacker to train: random, optimal, or agent.
-	`--beacon_type`: Type of beacon model to use: truth, agent, or beacon_strategy.
-	`--update_freq`: Frequency of model updates per episode.
-  `--seed`: Random seed for reproducibility.

##### Environment Setup

-	`--a_control_size`: Number of individuals in the attacker’s control group (default: 50).
-	`--b_control_size`: Number of individuals in the beacon’s control group (default: 50).
-	`--gene_size`: Size of the gene dataset.
-	`--beacon_size`: Population size for the beacon dataset.
-	`--victim_prob`: Probability that a victim is included in the beacon dataset.
-	`--max_queries`: Maximum number of queries per episode.
-	`--binary`: Specify if queries are binary (default: False).
-	`--user_risk`: Risk level for end-user queries (default: 0.2).

---

### Evaluating Trained Models

To simulate attacks and defenses, specify desired options such as beacon and attacker types, maximum queries, and user risk level.

```shell
python simulate.py --data /path/to/data --evaluate True 
```

#### Key Arguments
- `--attacker_type`: `"random"`, `"optimal"`, or `"agent"`.
- `--beacon_type`: `"truth"`, `"agent"`, `"baseline"`, etc.
- `--user_risk`: Controls the level of risk in queries for random attacker.
-	`--max_queries`: Maximum number of queries per episode.

---

## Citations

---

## License

- **[CC BY-NC-SA 2.0](https://creativecommons.org/licenses/by-nc-sa/2.0/)**
- © 2024 Beacon Defender Framework.
- **For commercial use, please contact.