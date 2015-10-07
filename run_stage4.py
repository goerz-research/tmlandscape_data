#!/usr/bin/env python
import sys
import run_stage3
run_stage3.STAGE = 'stage4'
from clusterjob import Job
if __name__ == "__main__":
    Job.default_backend = 'slurm'
    Job.cache_folder='./.clusterjob_cache/'+run_stage3.STAGE+'/'
    Job.default_sleep_interval = 180


if __name__ == "__main__":
    sys.exit(run_stage3.main())
