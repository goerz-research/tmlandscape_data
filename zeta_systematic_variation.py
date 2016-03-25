import numpy as np
from collections import OrderedDict
from QDYN.linalg import vectorize
import pandas as pd
import os
from os.path import join
from closest_targets import get_cpus, make_threadpool_map
from functools import partial
from oct_propagate import propagate
from notebook_utils import pulse_config_compat, avg_freq, max_freq_delta


def worker(args):
    rf, pulse_json = args
    U = propagate(rf, pulse_json, rwa=True, force=True, keep=None)
    return U


def systematic_variation(rf, pulse0, vary, fig_of_merit, n_procs=None,
        _worker=None):
    """Make a table of the fig_of_merit(U) when the parameters
    in the analytical pulse `pulse0` are varied as specified in
    the dict `vary`. Use `n_procs` parallel processes"""
    if n_procs is None:
        n_procs = get_cpus()
    if _worker is None:
        _worker = worker
    vals = OrderedDict([])
    keys = list(vary.keys())
    grid = np.meshgrid(*[vary[k] for k in keys])
    N = 0
    for i, key in enumerate(keys):
        vals[key] = pd.Series(vectorize(grid[i]))
        N = len(vals[key])
    threadpool_map = make_threadpool_map(n_procs)
    pulse_files = []
    for i in range(N):
        pulse = pulse0.copy()
        for key in keys:
            pulse.parameters[key] = vals[key][i]
        w_d = avg_freq(pulse) # GHz
        w_max = max_freq_delta(pulse, w_d) # GHZ
        pulse.parameters['w_d'] = w_d
        pulse.nt = int(max(2000, 100 * w_max * pulse.T))
        pulse_json = "pulse_variation_%d.json" % (i+1)
        pulse.write(join(rf, pulse_json))
        pulse_files.append((rf, pulse_json))
    Us = threadpool_map(_worker, pulse_files)
    for runfolder, pulse_file in pulse_files:
        os.unlink(join(runfolder, pulse_file))
    vals['fig_of_merit'] = [fig_of_merit(U) for U in Us]
    return pd.DataFrame(vals).sort('fig_of_merit')

