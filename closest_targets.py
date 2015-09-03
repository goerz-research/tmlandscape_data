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


def get_closest_targets(runs):
    thread_pool_map = make_threadpool_map(get_cpus())
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
                if os.path.isfile(U_dat) and not os.path.isfile(U_closest_dat):
                    if target == 'PE':
                        PE_gate_files.append(U_dat)
                    else:
                        SQ_gate_files.append(U_dat)
        # Note: memoization does NOT work with parallel execution (because a
        # thread pool has no shared memory)
        # process SQ gates
        QDYN.gate2q.closest_SQ.clear()
        SQ_cache_file = os.path.join(runs, "closest_SQ_%s.cache"%stage)
        QDYN.gate2q.closest_SQ.load(SQ_cache_file)
        thread_pool_map(get_closest_SQ, SQ_gate_files[:5])
        QDYN.gate2q.closest_SQ.dump(SQ_cache_file)
        # process PE gates
        QDYN.gate2q.closest_PE.clear()
        PE_cache_file = os.path.join(runs, "closest_PE_%s.cache"%stage)
        QDYN.gate2q.closest_PE.load(PE_cache_file)
        thread_pool_map(get_closest_PE, PE_gate_files)
        QDYN.gate2q.closest_PE.dump(PE_cache_file)


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

def get_closest(gatefile, target):
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
            return
    with open(outfile, 'w') as out_fh:
        QDYN.io.print_matrix(U_closest, out=out_fh)


def main(argv=None):
    """Main routine"""
    from optparse import OptionParser
    if argv is None:
        argv = sys.argv
    arg_parser = OptionParser(
    usage = "%prog [options] <runs>",
    description = __doc__)
    options, args = arg_parser.parse_args(argv)
    try:
        runs = args[1]
    except IndexError:
        arg_parser.error("Missing arguments")
    get_closest_targets(runs)

if __name__ == "__main__":
    sys.exit(main())
