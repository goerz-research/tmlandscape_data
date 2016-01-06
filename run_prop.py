#!/usr/bin/env python
"""Run the propagation stage.

Calls ./run_oct.py --prop-only for any runfolder that has 'stage_prop' (or
whatever is given as --stage-folder) in its path, inside RUNS (as prepared by
./select_for_prop.py).

Assumes that the runfolders contain the files config, pulse.dat.

All files that are generated during propagation will be kept in the runfolder.
"""
import sys
import run_stage3
run_stage3.STAGE = 'stage_prop' # default --stage-folder
run_stage3.PROP_ONLY = True
run_stage3.__doc__ = __doc__
from clusterjob import Job
if __name__ == "__main__":
    Job.default_backend = 'slurm'
    #Job.cache_folder='./.clusterjob_cache/'+run_stage3.STAGE+'/'
    Job.default_sleep_interval = 180


if __name__ == "__main__":
    sys.exit(run_stage3.main())
