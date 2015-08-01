#!/usr/bin/env python
from notebook_utils import find_folders
import logging
logging.basicConfig(level=logging.ERROR)
import sys
import time
import hashlib
import os
from textwrap import dedent
from clusterjob import Job
Job.default_remote = 'kcluster'
Job.default_backend = 'slurm'
Job.default_rootdir = '~/jobs/ConstrainedTransmon'
Job.default_opts['queue'] = 'AG-KOCH'
Job.cache_folder='./.clusterjob_cache/stage2/'

from run_stage1 import jobscript, epilogue, split_seq

def prologue(runs):
    assert runs.startswith("./"), "runs must be to current directory"
    runs = runs[2:]         # strip leading ./
    if runs.endswith(r'/'): # strip trailing slash
        runs = runs[:-1]
    prologue = dedent(r'''
    #!/bin/bash
    ssh {{remote}} mkdir -p {{rootdir}}/{runs}
    rsync -av ./{runs}/ {{remote}}:{{rootdir}}/{runs}
    '''.format(runs=runs))
    return prologue


def main(argv=None):
    """Run stage 2"""
    from optparse import OptionParser
    if argv is None:
        argv = sys.argv
    arg_parser = OptionParser(
    usage = "usage: %prog [options] RUNS",
    description = __doc__)
    arg_parser.add_option(
        '--rwa', action='store_true', dest='rwa',
        default=False, help="Perform all calculations in the RWA.")
    arg_parser.add_option(
        '--parallel', action='store', dest='parallel', type=int,
        default=3, help="Number of parallel processes per job [3]")
    arg_parser.add_option(
        '--jobs', action='store', dest='jobs', type=int,
        default=10, help="Number of jobs [10]")
    options, args = arg_parser.parse_args(argv)
    try:
        runs = os.path.join('.', os.path.normpath(args[1]))
    except IndexError:
        arg_parser.error("You must give RUNS")
    if not os.path.isdir(runs):
        arg_parser.error("RUNS must be a folder (%s)" % runs)
    if not runs.startswith(r'./'):
        arg_parser.error('RUNS must be relative to current folder, '
                         'e.g. ./runs')
    rwa = ''
    if options.rwa:
        rwa = '--rwa'
    submitted = []
    jobs = []
    job_ids = {}

    with open("stage2.log", "a") as log:
        log.write("%s\n" % time.asctime())
        for folder in find_folders(runs, 'stage2'):
            for subfolder in os.listdir(folder):
                runfolder = os.path.join(folder, subfolder)
                if not os.path.isdir(runfolder):
                    continue
                pulse_opt = os.path.join(runfolder, 'pulse_opt.json')
                if os.path.isfile(pulse_opt):
                    continue
                command = './stage2_simplex.py {rwa} {runfolder}'\
                          .format(rwa=rwa, runfolder=runfolder)
                jobs.append(command)
        for i_job, commands in enumerate(split_seq(jobs, options.jobs)):
            if len(commands) == 0:
                continue
            jobname = '%s_stage2_%02d' % (
                      runs.replace('.','').replace('/',''), i_job+1)
            job = Job(jobscript=jobscript(commands, options.parallel),
                    jobname=jobname, workdir='.', time='48:00:00',
                    nodes=1, threads=4*options.parallel,
                    mem=1000, stdout='%s-%%j.out'%jobname,
                    prologue=prologue(runs), epilogue=epilogue(runs))
            cache_id = '%s_%s' % (
                        jobname, hashlib.sha256(str(argv)).hexdigest())
            submitted.append(job.submit(cache_id=cache_id))
            job_ids[submitted[-1].job_id] = jobname
            log.write("Submitted %s to cluster as ID %s\n"%(
                       jobname, submitted[-1].job_id))

    for job in submitted:
        job.wait()
        if not job.successful():
            print "job '%s' did not finish successfully" % job_ids[job.job_id]

if __name__ == "__main__":
    sys.exit(main())
