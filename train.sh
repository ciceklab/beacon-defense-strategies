#!/bin/bash

eval "$(conda shell.bash hook)"

conda activate /mnt/sobhan/cpi


CUDA_VISIBLE_DEVICES=2 nohup python main.py --train beacon --attacker_type optimal --beacon_type agent --gene_size 1000000 --max_queries 100 --results-dir ./results/train > ./logs/beacon_gene_att_more_optimal.txt &