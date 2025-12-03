# -*- coding: UTF-8 -*-
from samplers.random_iter import RandomIterSampler
from samplers.uncertainty import UncertaintySampler
from samplers.typi_clust import TypiClustSampler
from samplers.uncertainty_diversity import UncertaintyDiversitySampler
from samplers.tcud import TypiClustUncertaintyDiversitySampler
from samplers.kmeans_ppp import KMeansPPPSampler            
from samplers.coreset import CoreSetSampler
from samplers.coreset_prob import CoreSetProbSampler
from samplers.typicore import TypiCoreSampler
from samplers.typi_kmeans import TypiKmeansSampler

samplers = {
    'random_iter': RandomIterSampler,
    'uncertainty': UncertaintySampler,
    'typi_clust': TypiClustSampler,
    'uncertainty_div': UncertaintyDiversitySampler,
    'tcud': TypiClustUncertaintyDiversitySampler,
    'kmeans_ppp': KMeansPPPSampler,
    'coreset': CoreSetSampler,
    'coreset_prob': CoreSetProbSampler,
    'typi_kmeans': TypiKmeansSampler,
    'typicore': TypiCoreSampler
}
