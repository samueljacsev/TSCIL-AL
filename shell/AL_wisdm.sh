#!/bin/bash

python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler random_iter & wait;\
python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler typi_clust & wait;\

# python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type margin & wait;\
# python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy & wait;\
# python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type lc & wait;\

python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler uncertainty --uncert_type margin & wait;\
python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler uncertainty --uncert_type entropy & wait;\
python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler uncertainty --uncert_type lc & wait;\

#python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler tcud --uncertainty_type lc & wait;\


# ER
python main_tune.py --data wisdm --encoder CNN --agent ER --norm BN --sampler random_iter & wait;\
python main_tune.py --data wisdm --encoder CNN --agent ER --norm BN --sampler typi_clust & wait;\

python main_tune.py --data wisdm --encoder CNN --agent ER --norm BN --sampler uncertainty --uncert_type margin & wait;\
python main_tune.py --data wisdm --encoder CNN --agent ER --norm BN --sampler uncertainty --uncert_type entropy & wait;\
python main_tune.py --data wisdm --encoder CNN --agent ER --norm BN --sampler uncertainty --uncert_type lc & wait;\