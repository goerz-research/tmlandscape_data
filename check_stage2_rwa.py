#!/usr/bin/env python
"""Check that the RWA is valid, by propagating the pulses obtained from the
simplex optimization in the LAB frame
"""
import os
import time
import sys
import logging
import hashlib
logging.basicConfig(level=logging.ERROR)
from clusterjob import Job
if __name__ == "__main__":
    Job.default_backend = 'slurm'
    Job.cache_folder='./.clusterjob_cache/check_stage2_rwa/'
    Job.default_sleep_interval = 180

from run_stage1 import split_seq, jobscript
from run_stage2 import prologue, epilogue
from notebook_utils import get_stage2_table


def get_jobs(runs):
    """Generate a list of calls to ./prop_lab.py for "interesting" subfolders
    in the given 'runs' root"""
    jobs = []
    stage2_table = get_stage2_table(runs)
    (__, t_PE), (__, t_SQ) = stage2_table.groupby('target', sort=True)
    for runfolder in t_PE[(t_PE['max loss']<0.1) & (t_PE['C']>0.9)].index:
        if not os.path.isfile(os.path.join(runfolder, 'U_LAB.dat')):
            jobs.append('./prop_lab.py %s' % runfolder)
    for runfolder in t_SQ[(t_SQ['max loss']<0.1) & (t_SQ['C']<0.1)].index:
        if not os.path.isfile(os.path.join(runfolder, 'U_LAB.dat')):
            jobs.append('./prop_lab.py %s' % runfolder)
    return jobs


def main(argv=None):
    """Run RWA check"""
    from optparse import OptionParser
    if argv is None:
        argv = sys.argv
    arg_parser = OptionParser(
    usage = "usage: %prog [options] RUNS",
    description = __doc__)
    arg_parser.add_option(
        '--parallel', action='store', dest='parallel', type=int,
        default=1, help="Number of parallel processes per job [1]")
    arg_parser.add_option(
        '--jobs', action='store', dest='jobs', type=int,
        default=10, help="Number of jobs [10]")
    arg_parser.add_option(
        '--local', action='store_true', dest='local',
        default=False, help="Submit all jobs to a SLURM cluster running "
        "directly on the local workstation")
    arg_parser.add_option(
        '-n', action='store_true', dest='dry_run',
        help="Perform a dry run")
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
    submitted = []
    jobs = get_jobs(runs)
    job_ids = {}

    if not options.local:
        Job.default_remote = 'kcluster'
        Job.default_opts['queue'] = 'AG-KOCH'
        Job.default_rootdir = '~/jobs/ConstrainedTransmon'

    with open("check_stage2_rwa.log", "a") as log:
        log.write("%s\n" % time.asctime())
        for i_job, commands in enumerate(split_seq(jobs, options.jobs)):
            if len(commands) == 0:
                continue
            jobname = 'check_stage2_rwa_%02d' % (i_job+1)
            if options.local:
                prologue_commands = None
                epilogue_commands = None
            else:
                prologue_commands = prologue(runs)
                epilogue_commands = epilogue(runs)
            job = Job(jobscript=jobscript(commands, options.parallel),
                    jobname=jobname, workdir='.', time='48:00:00',
                    nodes=1, threads=4*options.parallel,
                    mem=4000, stdout='%s-%%j.out'%jobname,
                    prologue=prologue_commands, epilogue=epilogue_commands)
            cache_id = '%s_%s' % (
                        jobname, hashlib.sha256(str(argv)).hexdigest())
            if options.dry_run:
                print "======== JOB %03d ========" % (i_job + 1)
                print job
                print "========================="
            else:
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
