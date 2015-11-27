#!/usr/bin/env python
"""Run stage3 (optimization)

Calls ./run_oct.py for any found runfolder that has 'stage3' in its path,
inside RUNS (as prepared by ./select_for_stage3.py).

Assumes that the runfolders contain the files config, pulse.guess.

Ceates the new files in the runfolders:
oct_iters.dat  oct.log  prop.log  pulse.dat  U.dat

If all the output files already exist, the runfolder is skipped unless
the --continue option is given.
"""
from notebook_utils import find_folders
import logging
logging.basicConfig(level=logging.ERROR)
import sys
import time
import hashlib
import os
PROP_ONLY = False
STAGE = 'stage3'
from clusterjob import Job
if __name__ == "__main__":
    Job.default_backend = 'slurm'
    Job.cache_folder='./.clusterjob_cache/'+STAGE+'/'
    Job.default_sleep_interval = 180
OUTFILES = ['oct_iters.dat', 'pulse.dat', 'U.dat']

from run_stage1 import jobscript, epilogue, split_seq
from run_stage2 import prologue


def main(argv=None):
    from optparse import OptionParser
    if argv is None:
        argv = sys.argv
    arg_parser = OptionParser(
    usage = "usage: %prog [options] RUNS",
    description = __doc__)
    arg_parser.add_option(
        '--rwa', action='store_true', dest='rwa',
        default=False, help="Perform all calculations in the RWA.")
    if not PROP_ONLY:
        arg_parser.add_option(
            '--continue', action='store_true', dest='cont',
            default=False, help="Continue optimization from aborted OCT. "
            "If not given, OCT will be skipped if all output files exist "
            "already.")
    arg_parser.add_option(
        '--parallel', action='store', dest='parallel', type=int,
        default=12, help="Number of parallel processes per job [12]")
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
    rwa = ''
    if options.rwa:
        rwa = '--rwa'
    submitted = []
    jobs = []
    job_ids = {}

    if not options.local:
        Job.default_remote = 'kcluster'
        Job.default_opts['queue'] = 'AG-KOCH'
        Job.default_rootdir = '~/jobs/ConstrainedTransmon'

    with open(os.path.join(runs, STAGE+".log"), "a") as log:
        log.write("%s\n" % time.asctime())
        for folder in find_folders(runs, STAGE):
            for subfolder in os.listdir(folder):
                runfolder = os.path.join(folder, subfolder)
                call_run_oct = False
                for file in OUTFILES:
                    if not os.path.isfile(os.path.join(runfolder, file)):
                        call_run_oct = True
                if not PROP_ONLY:
                    if options.cont:
                            call_run_oct = True
                if call_run_oct:
                    log.write("scheduled %s for OCT\n" % runfolder)
                    if PROP_ONLY:
                        command = './run_oct.py --prop-only --keep {rwa} {runfolder}'\
                                .format(rwa=rwa, runfolder=runfolder)
                    else:
                        command = './run_oct.py --continue {rwa} {runfolder}'\
                                .format(rwa=rwa, runfolder=runfolder)
                    jobs.append(command)
                else:
                    log.write("skipped %s (output complete)\n" % runfolder)
        for i_job, commands in enumerate(split_seq(jobs, options.jobs)):
            if len(commands) == 0:
                continue
            jobname = ('%s_%s_%02d') % (
                      runs.replace('.','').replace('/',''), STAGE, i_job+1)
            if options.local:
                prologue_commands = None
                epilogue_commands = None
            else:
                prologue_commands = prologue(runs)
                epilogue_commands = epilogue(runs)
            job = Job(jobscript=jobscript(commands, options.parallel),
                    jobname=jobname, workdir='.', time='24:00:00',
                    nodes=1, threads=options.parallel,
                    mem=1000*options.parallel, stdout='%s-%%j.out'%jobname,
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
