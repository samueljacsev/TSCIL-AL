# python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler random_iter & wait; \
# python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy & wait; 

python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler random_iter & wait; \
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy & wait; 