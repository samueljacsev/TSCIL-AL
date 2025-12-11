
# DSA with OOD methods
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method msp & wait;\
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method mahalanobis & wait;\
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method energy & wait;\

python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method msp & wait;\
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method mahalanobis & wait;\
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method energy & wait;\

# DSA without OOD methods
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler typi_clust & wait;\
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy & wait;\
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler random_iter & wait;\


# grabmyo without ood methods
python main_tune.py --data grabmyo --encoder CNN --agent ASER --norm BN --sampler typi_clust & wait;\


# har without ood methods
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler typi_clust & wait;\

# uwave without ood methods
python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler typi_clust & wait;\


# wisdmart without ood methods
python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler typi_clust & wait;\



python main_tune.py --data grabmyo --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method msp; `
python main_tune.py --data grabmyo --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method mahalanobis; `
python main_tune.py --data grabmyo --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method energy; `
python main_tune.py --data grabmyo --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method msp; `
python main_tune.py --data grabmyo --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method mahalanobis; `
python main_tune.py --data grabmyo --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method energy; `
python main_tune.py --data grabmyo --encoder CNN --agent ASER --norm BN --sampler typi_clust; `
python main_tune.py --data grabmyo --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy



python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method msp; `
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method mahalanobis; `
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method energy; `

python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method msp; `
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method mahalanobis; `
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method energy; `
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy; `
python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method msp; `
python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method mahalanobis; `
python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method energy; `
python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy


python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler typi_clust; `
# python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy


python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler random_iter; `
python main_tune.py --data grabmyo --encoder CNN --agent ASER --norm BN --sampler typi_clust; `
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler typi_clust; `
python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler typi_clust; `
python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler typi_clust

####################################################################################################################################################################
python main_tune.py --data grabmyo --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method msp; `
python main_tune.py --data grabmyo --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method mahalanobis; `
python main_tune.py --data grabmyo --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method energy; `
python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method msp; `
python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method mahalanobis; `
python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method energy; `
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method msp; `
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method mahalanobis; `
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method energy; `
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method msp; `
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method mahalanobis; `
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method energy; `
python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method msp; `
python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method mahalanobis; `
python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method energy; `
python main_tune.py --data grabmyo --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy; `
python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy; `
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy; `
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy; `
python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy; `




########################################################################## HAR  ##########################################################################################

python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method msp  --shuffle_ratio 0.5; `
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method mahalanobis --shuffle_ratio 0.5; `
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method energy  --shuffle_ratio 0.5; `
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --shuffle_ratio 0.5; `
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method msp --shuffle_ratio 0.5; `
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method mahalanobis --shuffle_ratio 0.5; `
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method energy --shuffle_ratio 0.5; `
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler typi_clust --shuffle_ratio 0.5; `
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method msp  --shuffle_ratio 0.35; `
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method mahalanobis --shuffle_ratio 0.35; `
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method energy  --shuffle_ratio 0.35; `
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --shuffle_ratio 0.35; `
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method msp --shuffle_ratio 0.35; `
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method mahalanobis --shuffle_ratio 0.35; `
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method energy --shuffle_ratio 0.35; `
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler typi_clust --shuffle_ratio 0.35; `
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method msp  --shuffle_ratio 0.2; `
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method mahalanobis --shuffle_ratio 0.2; `
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method energy  --shuffle_ratio 0.2; `
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --shuffle_ratio 0.2; `
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method msp --shuffle_ratio 0.2; `
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method mahalanobis --shuffle_ratio 0.2; `
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method energy --shuffle_ratio 0.2; `
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler typi_clust --shuffle_ratio 0.2; `


########################################################################## UWave  ##########################################################################################

python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method msp  --shuffle_ratio 0.5; `
python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method mahalanobis --shuffle_ratio 0.5; `
python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method energy  --shuffle_ratio 0.5; `
python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --shuffle_ratio 0.5; `
python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method msp --shuffle_ratio 0.5; `
python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method mahalanobis --shuffle_ratio 0.5; `
python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method energy --shuffle_ratio 0.5; `
python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler typi_clust --shuffle_ratio 0.5; `
python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method msp  --shuffle_ratio 0.35; `
python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method mahalanobis --shuffle_ratio 0.35; `
python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method energy  --shuffle_ratio 0.35; `
python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --shuffle_ratio 0.35; `
python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method msp --shuffle_ratio 0.35; `
python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method mahalanobis --shuffle_ratio 0.35; `
python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method energy --shuffle_ratio 0.35; `
python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler typi_clust --shuffle_ratio 0.35; `
python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method msp  --shuffle_ratio 0.2; `
python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method mahalanobis --shuffle_ratio 0.2; `
python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method energy  --shuffle_ratio 0.2; `
python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --shuffle_ratio 0.2; `
python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method msp --shuffle_ratio 0.2; `
python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method mahalanobis --shuffle_ratio 0.2; `
python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method energy --shuffle_ratio 0.2; `
python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler typi_clust --shuffle_ratio 0.2; `
########################################################################## Wisdm  ##########################################################################################
python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method msp  --shuffle_ratio 0.5; `
python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method mahalanobis --shuffle_ratio 0.5; `
python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method energy  --shuffle_ratio 0.5; `
python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --shuffle_ratio 0.5; `
python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method msp --shuffle_ratio 0.5; `
python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method mahalanobis --shuffle_ratio 0.5; `
python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method energy --shuffle_ratio 0.5; `
python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler typi_clust --shuffle_ratio 0.5; `
python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method msp  --shuffle_ratio 0.35; `
python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method mahalanobis --shuffle_ratio 0.35; `
python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method energy  --shuffle_ratio 0.35; `
python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --shuffle_ratio 0.35; `
python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method msp --shuffle_ratio 0.35; `
python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method mahalanobis --shuffle_ratio 0.35; `
python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method energy --shuffle_ratio 0.35; `
python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler typi_clust --shuffle_ratio 0.35; `
python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method msp  --shuffle_ratio 0.2; `
python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method mahalanobis --shuffle_ratio 0.2; `
python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method energy  --shuffle_ratio 0.2; `
python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --shuffle_ratio 0.2; `
python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method msp --shuffle_ratio 0.2; `
python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method mahalanobis --shuffle_ratio 0.2; `
python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method energy --shuffle_ratio 0.2; `
python main_tune.py --data wisdm --encoder CNN --agent ASER --norm BN --sampler typi_clust --shuffle_ratio 0.2; `
########################################################################## Grabmyo  ##########################################################################################


python main_tune.py --data grabmyo --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method msp  --shuffle_ratio 0.5; `
python main_tune.py --data grabmyo --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method mahalanobis --shuffle_ratio 0.5; `
python main_tune.py --data grabmyo --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method energy  --shuffle_ratio 0.5; `
python main_tune.py --data grabmyo --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --shuffle_ratio 0.5; `
python main_tune.py --data grabmyo --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method msp --shuffle_ratio 0.5; `
python main_tune.py --data grabmyo --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method mahalanobis --shuffle_ratio 0.5; `
python main_tune.py --data grabmyo --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method energy --shuffle_ratio 0.5; `
python main_tune.py --data grabmyo --encoder CNN --agent ASER --norm BN --sampler typi_clust --shuffle_ratio 0.5; `
python main_tune.py --data grabmyo --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method msp  --shuffle_ratio 0.35; `
python main_tune.py --data grabmyo --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method mahalanobis --shuffle_ratio 0.35; `
python main_tune.py --data grabmyo --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method energy  --shuffle_ratio 0.35; `
python main_tune.py --data grabmyo --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --shuffle_ratio 0.35; `
python main_tune.py --data grabmyo --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method msp --shuffle_ratio 0.35; `
python main_tune.py --data grabmyo --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method mahalanobis --shuffle_ratio 0.35; `
python main_tune.py --data grabmyo --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method energy --shuffle_ratio 0.35; `
python main_tune.py --data grabmyo --encoder CNN --agent ASER --norm BN --sampler typi_clust --shuffle_ratio 0.35; `
python main_tune.py --data grabmyo --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method msp  --shuffle_ratio 0.2; `
python main_tune.py --data grabmyo --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method mahalanobis --shuffle_ratio 0.2; `
python main_tune.py --data grabmyo --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method energy  --shuffle_ratio 0.2; `
python main_tune.py --data grabmyo --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --shuffle_ratio 0.2; `
python main_tune.py --data grabmyo --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method msp --shuffle_ratio 0.2; `
python main_tune.py --data grabmyo --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method mahalanobis --shuffle_ratio 0.2; `
python main_tune.py --data grabmyo --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method energy --shuffle_ratio 0.2; `
python main_tune.py --data grabmyo --encoder CNN --agent ASER --norm BN --sampler typi_clust --shuffle_ratio 0.2; `

########################################################################## Dailysports  ##########################################################################################

python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method msp  --shuffle_ratio 0.5; `
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method mahalanobis --shuffle_ratio 0.5; `
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method energy  --shuffle_ratio 0.5; `
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --shuffle_ratio 0.5; `
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method msp --shuffle_ratio 0.5; `
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method mahalanobis --shuffle_ratio 0.5; `
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method energy --shuffle_ratio 0.5; `
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler typi_clust --shuffle_ratio 0.5; `
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method msp  --shuffle_ratio 0.35; `
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method mahalanobis --shuffle_ratio 0.35; `
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method energy  --shuffle_ratio 0.35; `
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --shuffle_ratio 0.35; `
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method msp --shuffle_ratio 0.35; `
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method mahalanobis --shuffle_ratio 0.35; `
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method energy --shuffle_ratio 0.35; `
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler typi_clust --shuffle_ratio 0.35; `
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method msp  --shuffle_ratio 0.2; `
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method mahalanobis --shuffle_ratio 0.2; `
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method energy  --shuffle_ratio 0.2; `
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --shuffle_ratio 0.2; `
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method msp --shuffle_ratio 0.2; `
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method mahalanobis --shuffle_ratio 0.2; `
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler typi_clust --ood_method energy --shuffle_ratio 0.2; `
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler typi_clust --shuffle_ratio 0.2; `


########################################################################### NEW OOD  ##########################################################################################

python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method knn  --shuffle_ratio 0.2; `
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method cosine --shuffle_ratio 0.2; `
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --shuffle_ratio 0.2; `

python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method knn  --shuffle_ratio 0.2; `
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method cosine --shuffle_ratio 0.2; `
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --shuffle_ratio 0.2; `

python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method knn  --shuffle_ratio 0.2; `
python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method cosine --shuffle_ratio 0.2; `
python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --shuffle_ratio 0.2; `


python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler random_iter --shuffle_ratio 0.2; `


########################################################################### selection budget benchmark  ##########################################################################################
##### 0.1 ####### 0.2 ######### 0.3 ########
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method knn  --shuffle_ratio 0.2; `
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method cosine --shuffle_ratio 0.2; `
python main_tune.py --data dailysports --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --shuffle_ratio 0.2; `

python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method knn  --shuffle_ratio 0.2; `
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method cosine --shuffle_ratio 0.2; `
python main_tune.py --data har --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --shuffle_ratio 0.2; `

python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method knn  --shuffle_ratio 0.2; `
python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --ood_method cosine --shuffle_ratio 0.2; `
python main_tune.py --data uwave --encoder CNN --agent ASER --norm BN --sampler uncertainty_diversity --uncertainty_type entropy --shuffle_ratio 0.2; `