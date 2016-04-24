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
from clusterjob import JobScript
OUTFILES = ['oct_iters.dat', 'pulse.dat', 'U.dat']

from run_stage1 import jobscript, split_seq


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
    if PROP_ONLY:
        arg_parser.add_option(
            '--rho', action='store_true', dest='prop_rho',
            default=False, help="Do the propagation in Liouville space.")
        arg_parser.add_option(
            '--debug', action='store_true', dest='debug',
            default=False, help="Enable debugging output")
        arg_parser.add_option(
            '--threads', action='store_true', dest='threads',
            default=False, help="Use OpenMP threads in propagation")
        arg_parser.add_option(
            '--n_qubit', action='store', dest='n_qubit', type="int",
            default=None, help="Use the given number of qubit levels, "
            "instead of the number specified in the config file.")
        arg_parser.add_option(
            '--n_cavity', action='store', dest='n_cavity', type="int",
            default=None, help="Use the given number of cavity levels, "
            "instead of the number specified in the config file.")
        arg_parser.add_option(
            '--rho-pop-plot', action='store_true', dest='rho_pop_plot',
            default=False, help="In combination with --rho, "
            "produce a population plot")
    else:
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
        '--cluster-ini', action='store', dest='cluster_ini',
                    help="INI file from which to load clusterjob defaults")
    arg_parser.add_option(
        '--stage-folder', action='store', dest='stage_folder',
        default=STAGE, help="Name of stage folder. Alternative stage "
        "folder names may be used to explore different OCT strategies. "
        "Defaults to "+STAGE)
    arg_parser.add_option(
        '--pre-simplex', action='store', dest='formula',
        help="Run simplex pre-optimization before Krotov")
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
    pre_simplex = ''
    if options.formula is not None:
        pre_simplex = '--pre-simplex=%s' % options.formula
    if options.cluster_ini is not None:
        JobScript.read_defaults(options.cluster_ini)
    submitted = []
    jobs = []
    job_ids = {}

    with open(os.path.join(runs, options.stage_folder+".log"), "a") as log:
        log.write("%s\n" % time.asctime())
        for folder in find_folders(runs, options.stage_folder):
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
                        prop_opts = '--prop-only --keep'
                        if options.prop_rho:
                            prop_opts += " --prop-rho"
                        if options.rho_pop_plot:
                            prop_opts += " --rho-pop-plot"
                        if options.n_qubit is not None:
                            prop_opts += " --prop-n_qubit=%d" % options.n_qubit
                        if options.n_cavity is not None:
                            prop_opts += " --prop-n_cavity=%d" % options.n_cavity
                        if options.debug:
                            prop_opts += " --debug"
                        if options.threads:
                            prop_opts += " --threads"
                        command = './run_oct.py {prop_opts} {rwa} {pre_simplex} {runfolder}'\
                                .format(rwa=rwa, pre_simplex=pre_simplex,
                                        prop_opts=prop_opts,
                                        runfolder=runfolder)
                    else:
                        oct_opts = '--continue'
                        if options.debug:
                            oct_opts += " --debug"
                        command = './run_oct.py {oct_opts} {rwa} {pre_simplex} {runfolder}'\
                                .format(oct_opts=oct_opts, rwa=rwa, pre_simplex=pre_simplex,
                                        runfolder=runfolder)
                    jobs.append(command)
                else:
                    log.write("skipped %s (output complete)\n" % runfolder)
        for i_job, commands in enumerate(split_seq(jobs, options.jobs)):
            if len(commands) == 0:
                continue
            jobname = ('%s_%s_%02d') % (
                      runs.replace('.','').replace('/',''),
                      options.stage_folder, i_job+1)
            job = JobScript(body=jobscript(commands, options.parallel),
                            jobname=jobname, nodes=1, threads=options.parallel,
                            stdout='%s-%%j.out'%jobname)
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
