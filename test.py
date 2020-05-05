from __future__ import absolute_import

import os

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from got10k.experiments import *
    
from circulant_matrix_tracker import TrackerCSK

if __name__ == '__main__':
    tracker = TrackerCSK()

#  OTB测试
    root_dir = os.path.expanduser('~/data/OTB')
    root_dir = "/data2/OTB"
    e = ExperimentOTB(root_dir, version=2015)
    e.run(tracker)
    e.report([tracker.name])
    root_dir = "/data/VOT2019"
    e = ExperimentVOT(root_dir , version = 2019 , read_image = False)
    e.run(tracker)
    e.report([tracker.name])
