#!/usr/bin/env python
"""Find the closest PE/SQ gates"""

import sys
import os
import subprocess as sp
import multiprocessing
import QDYN
from notebook_utils import find_folders
import logging
logging.basicConfig(level=logging.INFO)

CPU_COUNT = 6

def get_cpus():
    """Return number of available cores, either SLURM-assigned cores or number
    of cores on the machine"""
    if CPU_COUNT is not None:
        return CPU_COUNT
    if 'SLURM_JOB_CPUS_PER_NODE' in os.environ:
        return int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    else:
        return multiprocessing.cpu_count()

def hostname():
    """Return the hostname"""
    import socket
    return socket.gethostname()


def make_threadpool_map(p):
    """Return a threadpool_map function using p parallel processes"""
    if p < 1:
        p = 1
    def threadpool_map(worker, jobs):
        """map worker routine over array of jobs, using a thread pool"""
        from multiprocessing import Pool
        if (p > 1):
            try:
                pool = Pool(processes=p)
                results = pool.map(worker, jobs)
            except sp.CalledProcessError, err:
                print "Encountered Error in thread pool"
                print err
        else: # single core: don't use ThreadPool
            results = []
            for job in jobs:
                results.append(worker(job))
        return results
    return threadpool_map


def get_closest_targets(runs, dry_run=False):
    thread_pool_map = make_threadpool_map(get_cpus())
    logger = logging.getLogger()
    for stage in ['stage2', 'stage3']:
        PE_gate_files = []
        SQ_gate_files = []
        folders = find_folders(runs, stage)
        for folder in folders:
            subfolders = os.listdir(folder)
            for subfolder in subfolders:
                target = 'PE'
                if subfolder.startswith("SQ"):
                    target = 'SQ'
                U_dat = os.path.join(folder, subfolder, 'U.dat')
                U_closest_dat = os.path.join(folder, subfolder,
                                             'U_closest_%s.dat'%target)
                if os.path.isfile(U_dat):
                    if not os.path.isfile(U_closest_dat):
                        if target == 'PE':
                            logger.debug("Require closest PE gate for %s",
                                         U_dat)
                            PE_gate_files.append(U_dat)
                        else:
                            logger.debug("Require closest SQ gate for %s",
                                         U_dat)
                            SQ_gate_files.append(U_dat)
                    else:
                        logger.debug("Skipping %s, closest gate already "
                                     "present", U_dat)
                else:
                    logger.debug("Skipping %s, does not exist", U_dat)
        if dry_run:
            for gate_file in SQ_gate_files:
                print("get_closests_SQ(%s)" % gate_file)
            for gate_file in PE_gate_files:
                print("get_closests_PE(%s)" % gate_file)
        else:
            # Note: memoization does NOT work with parallel execution (because
            # a thread pool has no shared memory)
            thread_pool_map(get_closest_SQ, SQ_gate_files)
            thread_pool_map(get_closest_PE, PE_gate_files)


# workers:

def get_closest_PE(gatefile):
    try:
        get_closest(gatefile, 'PE')
    except Exception as e:
        print("ERROR %s: %s"%(gatefile, e))


def get_closest_SQ(gatefile):
    try:
        get_closest(gatefile, 'SQ')
    except Exception as e:
        print("ERROR %s: %s"%(gatefile, e))


def get_closest(gatefile, target, raise_exceptions=False):
    logger = logging.getLogger()
    U = QDYN.gate2q.Gate2Q(file=gatefile)
    root, ext = os.path.splitext(gatefile)
    outfile = root + "_closest_%s"%target + ext
    if target == 'SQ':
        f_closest = QDYN.gate2q.closest_SQ
    else:
        f_closest = QDYN.gate2q.closest_PE
    try:
        U_closest = f_closest(U)
    except ValueError:
        try:
            U_closest = f_closest(U.closest_unitary())
        except ValueError:
            logger.warn("Could not find closest unitary for %s", gatefile)
            if raise_exceptions:
                raise
            else:
                return None
    with open(outfile, 'w') as out_fh:
        QDYN.io.print_matrix(U_closest, out=out_fh)
    return U_closest


def main(argv=None):
    """Main routine"""
    from optparse import OptionParser
    if argv is None:
        argv = sys.argv
    arg_parser = OptionParser(
    usage = "%prog [options] <runs>",
    description = __doc__)
    arg_parser.add_option(
        '--debug', action='store_true', dest='debug',
        default=False, help="Enable debugging output")
    arg_parser.add_option(
        '-n', action='store_true', dest='dry_run',
        default=False, help="Perform a dry run")
    options, args = arg_parser.parse_args(argv)
    logger = logging.getLogger()
    if options.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    try:
        runs = args[1]
    except IndexError:
        arg_parser.error("Missing arguments")
    get_closest_targets(runs, dry_run=options.dry_run)

if __name__ == "__main__":
    sys.exit(main())
