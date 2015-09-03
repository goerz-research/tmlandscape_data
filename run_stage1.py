#!/usr/bin/env python
"""Run stage 1 optimization"""
import logging
logging.basicConfig(level=logging.ERROR)
import os
import sys
import time
import hashlib
from textwrap import dedent
from clusterjob import Job
import numpy as np
from notebook_utils import PlotGrid
Job.default_remote = 'kcluster'
Job.default_backend = 'slurm'
Job.default_rootdir = '~/jobs/ConstrainedTransmon'
Job.default_opts['queue'] = 'AG-KOCH'
Job.cache_folder='./.clusterjob_cache/stage1/'


def generate_parameters(outfile):
    """Write a file of w_2, w_c parameters (to be called manually)"""
    w1 = 6.0
    w2_wc = []
    for w2 in [5.0, 5.5, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.29, 6.30, 6.31,
    6.35, 6.5, 6.58, 6.0, 6.62, 6.75, 7.0, 7.25, 7.5]:
        for wc in [5.0, 5.8, 6.1, 6.3, 6.6, 7.1, 7.6, 8.1, 8.6, 9.1, 10.1,
        11.1]:
            if (   (abs(w1-w2) <= 1.8)
                or (abs(wc-w2) <= 1.8)
                or (abs(wc-w1) <= 1.8)
            ):
                w2_wc.append((w2, wc))
    w2_wc.append((7.5, 11.1))
    preview_points(w2_wc)
    write_w2_wc(w2_wc, outfile)


def split_seq(seq, n_chunks):
    """Split the given sequence into n_chunks"""
    newseq = []
    splitsize = 1.0/n_chunks*len(seq)
    for i in range(n_chunks):
        newseq.append(seq[int(round(i*splitsize)):int(round((i+1)*splitsize))])
    return newseq


def preview_points(w2_wc):
    """Show a plot of the sampling points"""
    p = PlotGrid()
    p.n_cols = 1
    w2, wc = zip(*w2_wc)
    val = np.ones(len(w2))
    p.add_cell(np.array(w2), np.array(wc), val, vmin=0.0, vmax=1.0)
    p.plot()


def write_w2_wc(w2_wc, filename):
    """Write the given list of tuples to the given filename. Each tuple is the
    value of w_2 (right qubit frequency) and w_c (cavity frequency), in GHz.
    w_1 is fixed at 6.0 GHz"""
    with open(filename, 'w') as out_fh:
        out_fh.write("# w_1 = 6.0 GHz\n")
        out_fh.write("#%9s%10s\n" % ("w2 [GHz]", "wc [GHz]"))
        for (w2, wc) in w2_wc:
            out_fh.write("%10.5f%10.5f\n" % (w2, wc))


def read_w2_wc(filename):
    """Read two columns from the given filename, w_2 and w_c in GHz, and
    return a list of tuples (w2, wc)"""
    w2, wc = np.genfromtxt(filename, unpack=True)
    return zip(w2, wc)


def jobscript(commands, parallel):
    jobscript = dedent(r'''
    source /usr/share/Modules/init/bash
    module load intel/14.0.3
    export PREFIX={PREFIX}
    export PATH=$PREFIX/bin:$PATH
    export LD_LIBRARY_PATH=$PREFIX/lib:$LD_LIBRARY_PATH

    xargs -L1 -P{parallel} python -u  <<EOF
    '''.format(
        PREFIX=os.path.join(Job.default_rootdir, 'venv'),
        parallel=parallel
    ))
    for command in commands:
        jobscript += command + "\n"
    jobscript += "EOF\n"
    return jobscript


def epilogue(runs):
    assert runs.startswith("./"), "runs must be to current directory"
    runs = runs[2:]         # strip leading ./
    if runs.endswith(r'/'): # strip trailing slash
        runs = runs[:-1]
    epilogue = dedent(r'''
    #!/bin/bash
    mkdir -p ./{runs}
    rsync -av {{remote}}:{{rootdir}}/{runs}/ ./{runs}
    '''.format(runs=runs))
    return epilogue


def main(argv=None):
    """Run stage 1"""
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
        '--duration', action='store', dest='T', type=float,
        default=200.0, help="Gate duration, in ns [200]")
    arg_parser.add_option(
        '--parallel', action='store', dest='parallel', type=int,
        default=3, help="Number of parallel processes per job [3]")
    arg_parser.add_option(
        '--jobs', action='store', dest='jobs', type=int,
        default=10, help="Number of jobs [10]")
    arg_parser.add_option(
        '--params-file', action='store', dest='params_file',
        help="File from which to read w2, wc tuples.")
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
    w1 = 6.0
    with open("stage1.log", "a") as log:
        if options.params_file is None:
            arg_parser.error('the --params-file option must be given')
        else:
            w2_wc = read_w2_wc(options.params_file)
        for (w2, wc) in w2_wc:
            command = './pre_simplex_scan.py {rwa} {runs} {w2} {wc} {T}'\
                      .format(rwa=rwa, runs=runs, w2=w2, wc=wc, T=options.T)
            jobs.append(command)
        for i_job, commands in enumerate(split_seq(jobs, options.jobs)):
            if len(commands) == 0:
                continue
            jobname = 'stage1_%02d' % (i_job+1)
            job = Job(jobscript=jobscript(commands, options.parallel),
                    jobname=jobname, workdir='.', time='200:00:00',
                    nodes=1, threads=4*options.parallel,
                    mem=40000, stdout='%s-%%j.out'%jobname,
                    epilogue=epilogue(runs))
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

