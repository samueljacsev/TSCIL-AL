# -*- coding: UTF-8 -*-
from samplers.full import FullSampler
from samplers.random import RandomSampler
from samplers.random_iter import RandomIterSampler
from samplers.uncertainty import UncertaintySampler
from samplers.typi_clust import TypiClustSampler
from samplers.uncertainty_diversity import UncertaintyDiversitySampler

samplers = {
    'full': FullSampler,
    'random': RandomSampler,
    'random_iter': RandomIterSampler,
    'uncertainty': UncertaintySampler,
    'typi_clust': TypiClustSampler,
    'uncertainty_diversity': UncertaintyDiversitySampler
}
