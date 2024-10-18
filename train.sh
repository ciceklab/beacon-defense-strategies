#!/bin/bash

eval "$(conda shell.bash hook)"

conda activate /mnt/sobhan/cpi


CUDA_VISIBLE_DEVICES=3 nohup python main.py --train beacon --attacker_type optimal --beacon_type agent --gene_size 1000000 --max_queries 100 --results-dir ./results/train > ./logs/beacon_log.txt &
# CUDA_VISIBLE_DEVICES=2 nohup python main.py --train attacker --attacker_type agent --beacon_type truth --gene_size 100000 --max_queries 100 --results-dir ./results/train > ./logs/attacker_log .txt &
