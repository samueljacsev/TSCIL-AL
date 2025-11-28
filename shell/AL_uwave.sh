#!/bin/bash

################################################# ASER ##############################################################
python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler random_iter & wait;\

python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler uncertainty --uncert_type margin & wait;\
python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler uncertainty --uncert_type entropy & wait;\
python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler uncertainty --uncert_type lc & wait;\

# python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncert_type margin & wait;\
# python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncert_type entropy & wait;\
# python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncert_type lc & wait;\

python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler typi_clust & wait;\

################################################# ER ##############################################################

python main_tune.py --data uwave --encoder CNN --agent ER --norm BN --sampler random_iter & wait;\

python main_tune.py --data uwave --encoder CNN --agent ER --norm BN --sampler uncertainty --uncert_type margin & wait;\
python main_tune.py --data uwave --encoder CNN --agent ER --norm BN --sampler uncertainty --uncert_type entropy & wait;\
python main_tune.py --data uwave --encoder CNN --agent ER --norm BN --sampler uncertainty --uncert_type lc & wait;\

# python main_tune.py --data uwave --encoder CNN --agent ER --norm BN --sampler uncertainty_diversity --uncert_type margin & wait;\
# python main_tune.py --data uwave --encoder CNN --agent ER --norm BN --sampler uncertainty_diversity --uncert_type entropy & wait;\
# python main_tune.py --data uwave --encoder CNN --agent ER --norm BN --sampler uncertainty_diversity --uncert_type lc & wait;\

python main_tune.py --data uwave --encoder CNN --agent ER --norm BN --sampler typi_clust & wait;\
