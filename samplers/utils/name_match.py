# -*- coding: UTF-8 -*-
from samplers.random_iter import RandomIterSampler
from samplers.uncertainty import UncertaintySampler
from samplers.typi_clust import TypiClustSampler
from samplers.uncertainty_diversity import UncertaintyDiversitySampler         
from samplers.coreset import CoreSetSampler
from samplers.typicore import TypiCoreSampler

samplers = {
    'random_iter': RandomIterSampler,
    'uncertainty': UncertaintySampler,
    'typi_clust': TypiClustSampler,
    'uncertainty_div': UncertaintyDiversitySampler,
    'coreset': CoreSetSampler,
    'typicore': TypiCoreSampler
}
