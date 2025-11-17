#!/bin/bash

#python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler random & wait;\

########## random_iter ##########
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler random_iter & wait;\

########## uncertainty ##########
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler uncertainty --uncertainty_type margin & wait;\
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler uncertainty --uncertainty_type entropy & wait;\
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler uncertainty --uncertainty_type least_confidence & wait;\

########## uncertainty_diversity ##########
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type margin & wait;\
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy & wait;\
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type least_confidence & wait;\

########## typi_clust ##########
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler typi_clust & wait;\



