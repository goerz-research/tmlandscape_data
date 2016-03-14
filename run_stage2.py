#!/usr/bin/env python
from notebook_utils import find_folders
import logging
import sys
import time
import hashlib
import os
from textwrap import dedent
from clusterjob import JobScript
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
        default=4, help="Number of parallel processes per job [4]")
    arg_parser.add_option(
        '--jobs', action='store', dest='jobs', type=int,
        default=10, help="Number of jobs [10]")
    arg_parser.add_option(
        '--cluster-ini', action='store', dest='cluster_ini',
                    help="INI file from which to load clusterjob defaults")
    arg_parser.add_option(
        '--debug', action='store_true', dest='debug',
        default=False, help="Enable debugging output")
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
    debug = ''
    if options.debug:
        debug = '--debug'
    rwa = ''
    if options.rwa:
        rwa = '--rwa'
    if options.cluster_ini is not None:
        JobScript.read_defaults(options.cluster_ini)
    submitted = []
    jobs = []
    job_ids = {}

    with open(os.path.join(runs, "stage2.log"), "a") as log:
        log.write("%s\n" % time.asctime())
        for folder in find_folders(runs, 'stage2'):
            for subfolder in os.listdir(folder):
                runfolder = os.path.join(folder, subfolder)
                if not os.path.isdir(runfolder):
                    continue
                pulse_opt = os.path.join(runfolder, 'pulse_opt.json')
                if os.path.isfile(pulse_opt):
                    continue
                command = './stage2_simplex.py {debug} {rwa} {runfolder}'\
                          .format(debug=debug, rwa=rwa, runfolder=runfolder)
                jobs.append(command)
        for i_job, commands in enumerate(split_seq(jobs, options.jobs)):
            if len(commands) == 0:
                continue
            jobname = '%s_stage2_%02d' % (
                      runs.replace('.','').replace('/',''), i_job+1)
            if options.local:
                prologue_commands = None
                epilogue_commands = None
            else:
                prologue_commands = prologue(runs)
                epilogue_commands = epilogue(runs)
            job = JobScript(body=jobscript(commands, options.parallel),
                            jobname=jobname, nodes=1, ppn=options.parallel,
                            stdout='%s-%%j.out'%jobname,
                            prologue=prologue_commands,
                            epilogue=epilogue_commands)
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
    logging.basicConfig(level=logging.ERROR)
    sys.exit(main())
