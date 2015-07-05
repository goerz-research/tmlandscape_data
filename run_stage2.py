#!/usr/bin/env python
from notebook_utils import find_folders
import logging
logging.basicConfig(level=logging.ERROR)
import sys
import time
import os
from textwrap import dedent
from clusterjob import Job
Job.default_remote = 'kcluster'
Job.default_backend = 'slurm'
Job.default_rootdir = '~/jobs/ConstrainedTransmon'
Job.default_opts['queue'] = 'AG-KOCH'
Job.cache_folder='./.clusterjob_cache/stage2/'


def jobscript(runfolder):
    jobscript = dedent(r'''
    source /usr/share/Modules/init/bash
    module load intel/14.0.3
    export PREFIX={PREFIX}
    export PATH=$PREFIX/bin:$PATH
    export LD_LIBRARY_PATH=$PREFIX/lib:$LD_LIBRARY_PATH

    python -u ./stage2_simplex.py {runfolder}
    '''.format(
        PREFIX='$HOME/jobs/ConstrainedTransmon/venv',
        runfolder=runfolder)
    )
    return jobscript


def epilogue(runfolder):
    epilogue = dedent(r'''
    #!/bin/bash
    mkdir -p ./runs/
    rsync -av {remote}:{rootdir}/%s/ %s
    '''% (runfolder, runfolder))
    return epilogue


def prologue(runfolder):
    epilogue = dedent(r'''
    #!/bin/bash
    ssh {remote} mkdir -p {rootdir}/%s
    rsync -av %s/ {remote}:{rootdir}/%s
    '''% (runfolder, runfolder, runfolder))
    return epilogue


def main():
    """Run stage 2 optimization"""
    jobs = []
    with open("stage2.log", "a") as log:
        log.write("%s\n" % time.asctime())
        for folder in find_folders('./runs', 'stage2'):
                for subfolder in os.listdir(folder):
                    runfolder = os.path.join(folder, subfolder)
                    if not os.path.isdir(runfolder):
                        continue
                    pulse_opt = os.path.join(runfolder, 'pulse_opt.json')
                    if os.path.isfile(pulse_opt):
                        continue
                    filename = runfolder.replace(r'/', '_')
                    filename = filename.replace(r'.', '') + ".slr"
                    job = Job(jobscript=jobscript(runfolder), jobname='stage2',
                            workdir='.', time='90:00:00', nodes=1, threads=4,
                            mem=4000, stdout='%j.out', filename=filename,
                            epilogue=epilogue(runfolder),
                            prologue=prologue(runfolder))
                    jobs.append(job.submit(cache_id=filename))
                    log.write("Submitted %s to cluster as ID %s\n"%(
                            runfolder, jobs[-1].job_id))
    for job in jobs:
        job.wait()
        if not job.successful():
            print "job '%s' did not finish successfully" % runfolder


if __name__ == "__main__":
    sys.exit(main())
