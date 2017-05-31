import matplotlib
matplotlib.use("Agg")

import os
import numpy as np
import matplotlib.pyplot as plt

project_dir = os.path.expanduser("~")+"/Work/DensityLearner/"
analysis_dir = os.path.join(project_dir, "density_analysis/")

a_weights = np.load(analysis_dir+"a_weight_matrix.npz")['data']
b_weights = np.load(analysis_dir+"b_weight_matrix.npz")['data']

import IPython; IPython.embed(); raise SystemExit
