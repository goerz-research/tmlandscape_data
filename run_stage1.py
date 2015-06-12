#!/usr/bin/env python
import logging
logging.basicConfig(level=logging.ERROR)
import sys
import time
from textwrap import dedent
from clusterjob import Job
Job.default_remote = 'kcluster'
Job.default_backend = 'slurm'
Job.default_rootdir = '~/jobs/ConstrainedTransmon'
Job.default_opts['queue'] = 'AG-KOCH'
Job.cache_folder='./.clusterjob_cache/stage1/'


def jobscript(w2, wc):
    jobscript = dedent(r'''
    source /usr/share/Modules/init/bash
    module load intel/14.0.3
    export PREFIX={PREFIX}
    export PATH=$PREFIX/bin:$PATH
    export LD_LIBRARY_PATH=$PREFIX/lib:$LD_LIBRARY_PATH

    python -u ./pre_simplex_scan.py {w2} {wc}
    '''.format(
        PREFIX='$HOME/jobs/ConstrainedTransmon/venv',
        w2=str(w2), wc=str(wc)
    ))
    return jobscript


def epilogue(w2, wc):
    epilogue = dedent(r'''
    #!/bin/bash
    mkdir -p ./runs/
    rsync -av {remote}:{rootdir}/runs/w2_%dMHz_wc_%dMHz ./runs/
    ''' % (w2*1000, wc*1000))
    return epilogue


def main():
    """Run stage 1 optimization"""
    # run pre_simplex_scan.py for every parameter point
    # select best runs for stage 2
    jobs = []
    job_ids = {}
    with open("stage1.log", "a") as log:
        log.write("%s\n" % time.asctime())
        for w2 in [6.0, 6.1, 6.2, 6.35, 6.5, 6.75, 7.0, 7.25, 7.5]:
            for wc in [5.0, 5.8, 6.1, 6.3, 6.6, 7.1, 7.6, 8.1, 8.6, 9.1, 10.1,
                       11.1]:
                jobname = 'w2_%dMHz_wc_%dMHz_stage1' % (w2*1000, wc*1000)
                job = Job(jobscript=jobscript(w2, wc), jobname=jobname,
                        workdir='.', time='8:00:00', nodes=1, threads=12,
                        mem=24000, stdout='%s-%%j.out'%jobname,
                        epilogue=epilogue(w2, wc))
                jobs.append(job.submit(cache_id=jobname))
                job_ids[jobs[-1].job_id] = jobname
                log.write("Submitted %s to cluster as ID %s\n"%(
                          jobname, jobs[-1].job_id))
    for job in jobs:
        job.wait()
        if not job.successful():
            print "job '%s' did not finish successfully" % job_ids[job.job_id]



if __name__ == "__main__":
    sys.exit(main())
