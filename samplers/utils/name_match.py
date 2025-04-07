# -*- coding: UTF-8 -*-
from samplers.full import FullSampler
from samplers.random import RandomSampler
from samplers.random_iter import RandomIterSampler
from samplers.uncertainty import UncertaintySampler

samplers = {
    'full': FullSampler,
    'random': RandomSampler,
    'random_iter': RandomIterSampler,
    'uncertainty': UncertaintySampler,
}
