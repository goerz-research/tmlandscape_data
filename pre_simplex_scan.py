#!/usr/bin/env python
"""Scan over guess pulses for system with right qubit frequency w_2, and cavity
frequency w_c, given in GHz"""

import sys
import os
import subprocess as sp
import numpy as np
import multiprocessing
import re
import shutil
from textwrap import dedent
from random import random
import QDYN
import logging
import time
logging.basicConfig(level=logging.INFO)


def get_cpus():
    """Return number of available cores, either SLURM-assigned cores or number
    of cores on the machine"""
    if 'SLURM_JOB_CPUS_PER_NODE' in os.environ:
        return int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    else:
        return multiprocessing.cpu_count()

def hostname():
    """Return the hostname"""
    import socket
    return socket.gethostname()


def make_random_freq(w_1, w_2, w_c, alpha_1, alpha_2, low_freq_limit,
    sidebands=True):
    """Return a random_freq function for generating random frequencies in the
    region of the system-typical frequencies. The low_freq_limit is the maximum
    frequency at which the shape is allowed to oscillate.

    If sidebands is True, allow for sideband frequencies such as transitions
    010 -> 101
    """
    delta = abs(w_2 - w_1)
    intervals1 = []
    intervals1.append([w_1-1.2*abs(alpha_1), w_2+0.2*abs(alpha_2)])
    if sidebands:
        intervals1.append([w_c-1.2*delta, w_c+1.2*delta])
    # collapse overlapping intervals
    intervals = []
    for i, interval in enumerate(intervals1):
        if i == 0:
            intervals.append(interval)
        else:
            (a, b) = interval
            (a_prev, b_prev) = intervals[-1]
            if a > b_prev:
                intervals.append(interval)
            else:
                if a < a_prev:
                    intervals[-1][0] = a
                if b > b_prev:
                    intervals[-1][1] = b
    def random_freq(n, n_low_freq=0):
        """Return n "normal" (high) random frequencies, and n_low_freq low
        frequency components"""
        result = []
        for realization in xrange(n):
            total_width = 0.0
            for (a,b) in intervals:
                total_width += b-a
            r = total_width * random()
            for (a,b) in intervals:
                w = b-a
                if r > w:
                    r -= w
                else:
                    result.append(a + r)
                    break
        for realization in xrange(n_low_freq):
            result.append(low_freq_limit*random())
        return np.array(result)
    return random_freq


def runfolder_to_params(runfolder):
    """From the full path to the runfolder, extract and return
    (w_2, w_c, E0, pulse_label), where w_2 is the second qubit frequency in
    MHz, w_c is the cavity frequency in MHz, E0 is the pulse amplitude in
    MHz, and pulse_label is an indicator of the pulse structure for that
    run, e.g. '2freq_resonant'"""
    if runfolder.endswith(r'/'):
        runfolder = runfolder[:-1]
    parts = runfolder.split(os.path.sep)
    E0_str = parts.pop()
    if E0_str == 'field_free':
        E0 = 0.0
        pulse_label = E0_str
    else:
        E0 = int(E0_str[1:])
        pulse_label = parts.pop()
    parts.pop() # discard 'stage1'
    w2_wc_str = parts.pop()
    w2_wc_match = re.match(r'w2_(\d+)MHz_wc_(\d+)MHz', w2_wc_str)
    if w2_wc_match:
        w_2 = float(w2_wc_match.group(1))
        w_c = float(w2_wc_match.group(2))
    else:
        raise ValueError("Could not get w_2, w_c from %s" % w2_wc_str)
    return w_2, w_c, E0, pulse_label


def generate_runfolders(w2, wc):
    """Generate a set of runfolders, ready for propagation. Returns a list of
    runfolder paths. w2 and wc must be given in GHz"""
    from analytical_pulses import AnalyticalPulse
    logger = logging.getLogger(__name__)
    logger.debug("Entering generate_runfolders")
    T = 200.0 # ns
    nt = 200*11*100
    config = dedent(r'''
    tgrid: n = 1
    1 : t_start = 0.0, t_stop = {T}_ns, nt = {nt}

    pulse: n = 1
    1: type = file, filename = pulse.guess, id = 1,  &
       time_unit = ns, ampl_unit = MHz, is_complex = F

    misc: prop = newton, mass = 1.0

    user_ints: n_qubit = 5, n_cavity = 6

    user_strings: gate = CPHASE, J_T = SM

    user_logicals: prop_guess = T, dissipation = T

    user_reals: &
    w_c     = {w_c}_MHz, &
    w_1     = 6000.0_MHz, &
    w_2     = {w_2}_MHz, &
    w_d     = 0.0_MHz, &
    alpha_1 = -290.0_MHz, &
    alpha_2 = -310.0_MHz, &
    J       =   0.0_MHz, &
    g_1     = 70.0_MHz, &
    g_2     = 70.0_MHz, &
    n0_qubit  = 0.0, &
    n0_cavity = 0.0, &
    kappa   = 0.05_MHz, &
    gamma_1 = 0.012_MHz, &
    gamma_2 = 0.012_MHz, &
    '''.format(T=T, nt=nt, w_c=(float(wc)*1000.0), w_2=(float(w2)*1000.0)))
    runfolders = []
    random_freq = make_random_freq(w_1=6.0, w_2=w2, w_c=wc, alpha_1=0.29,
                                   alpha_2=0.31, low_freq_limit=0.02,
                                   sidebands=True)

    runfolder_root = 'runs/w2_%dMHz_wc_%dMHz/stage1' % (w2*1000, wc*1000)

    def runfolder_exists(runfolder):
        return (os.path.isfile(os.path.join(runfolder, 'config'))
            and os.path.isfile(os.path.join(runfolder, 'pulse.json')))

    def write_runfolder(runfolder, analytical_pulse, config):
        QDYN.shutil.mkdir(runfolder)
        with open(os.path.join(runfolder, 'config'), 'w') as config_fh:
            config_fh.write(config)
        analytical_pulse.write(os.path.join(runfolder, 'pulse.json'),
                               pretty=True)

    amplitudes = [10, 50, 100, 200, 300, 400, 500, 700, 900]

    # field-free
    pulse_label = 'field_free'
    runfolder = os.path.join(runfolder_root, pulse_label)
    if runfolder_exists(runfolder):
        logger.debug("%s exists (skipped)", runfolder)
    else:
        logger.debug("Generating %s", runfolder)
        pulse = AnalyticalPulse('field_free', T, nt,
                parameters={}, time_unit='ns', ampl_unit='MHz')
        write_runfolder(runfolder, pulse, config)
    runfolders.append(runfolder)

    # single-frequency (center)
    for E0 in amplitudes:
        pulse_label = '1freq_center'
        runfolder = os.path.join(runfolder_root, pulse_label, "E%03d"%E0)
        if runfolder_exists(runfolder):
            logger.debug("%s exists (skipped)", runfolder)
        else:
            logger.debug("Generating %s", runfolder)
            w_L = 0.5*(6.0 + w2)
            pulse = AnalyticalPulse('1freq', T, nt,
                    parameters={'E0': E0, 'T': T, 'w_L': w_L},
                    time_unit='ns', ampl_unit='MHz')
            write_runfolder(runfolder, pulse, config)
        runfolders.append(runfolder)

    # single-frequency (random)
    for realization in xrange(10):
        w_L = float(random_freq(1)[0])
        for E0 in amplitudes:
            pulse_label = '1freq_%d'%(realization+1)
            runfolder = os.path.join(runfolder_root, pulse_label, "E%03d"%E0)
            if runfolder_exists(runfolder):
                logger.debug("%s exists (skipped)", runfolder)
            else:
                logger.debug("Generating %s", runfolder)
                pulse = AnalyticalPulse('1freq', T, nt,
                        parameters={'E0': E0, 'T': T, 'w_L': w_L},
                        time_unit='ns', ampl_unit='MHz')
                write_runfolder(runfolder, pulse, config)
            runfolders.append(runfolder)

    # two-frequency (resonant)
    for E0 in amplitudes:
        pulse_label = '2freq_resonant'
        runfolder = os.path.join(runfolder_root, pulse_label, "E%03d"%E0)
        if runfolder_exists(runfolder):
            logger.debug("%s exists (skipped)", runfolder)
        else:
            logger.debug("Generating %s", runfolder)
            freq_1 = 6.0
            freq_2 = w2
            phi = 0.0
            a_1 = 1.0
            a_2 = 1.0
            pulse = AnalyticalPulse('2freq', T, nt,
                    parameters={'E0': E0, 'T': T, 'freq_1': freq_1,
                                'freq_2': freq_2, 'a_1': a_1, 'a_2': a_2,
                                'phi': phi},
                    time_unit='ns', ampl_unit='MHz')
            write_runfolder(runfolder, pulse, config)
        runfolders.append(runfolder)

    # two-frequency (random)
    for realization in xrange(10):
        freq_1, freq_2 = random_freq(2)
        phi = 2*random()
        a_1 = random()
        a_2 = random()
        for E0 in amplitudes:
            pulse_label = '2freq_%d'%(realization+1)
            runfolder = os.path.join(runfolder_root, pulse_label, "E%03d"%E0)
            if runfolder_exists(runfolder):
                logger.debug("%s exists (skipped)", runfolder)
            else:
                logger.debug("Generating %s", runfolder)
                pulse = AnalyticalPulse('2freq', T, nt,
                        parameters={'E0': E0, 'T': T, 'freq_1': freq_1,
                                    'freq_2': freq_2, 'a_1': a_1, 'a_2': a_2,
                                    'phi': phi},
                        time_unit='ns', ampl_unit='MHz')
                write_runfolder(runfolder, pulse, config)
            runfolders.append(runfolder)

    # five-frequency (random)
    for realization in xrange(10):
        freq = random_freq(3, n_low_freq=2)
        a = np.random.rand(5) - 0.5
        b = np.random.rand(5) - 0.5
        for E0 in amplitudes:
            pulse_label = '5freq_%d'%(realization+1)
            runfolder = os.path.join(runfolder_root, pulse_label, "E%03d"%E0)
            if runfolder_exists(runfolder):
                logger.debug("%s exists (skipped)", runfolder)
            else:
                logger.debug("Generating %s", runfolder)
                pulse = AnalyticalPulse('5freq', T, nt,
                        parameters={'E0': E0, 'T': T, 'freq': freq,
                                    'a': a, 'b': b},
                        time_unit='ns', ampl_unit='MHz')
                write_runfolder(runfolder, pulse, config)
            runfolders.append(runfolder)

    logger.debug("Finished generate_runfolders")
    return runfolders


def get_temp_runfolder(runfolder):
    """Return the path for an appropriate temporary runfolder (inside
    $SCRATCH_ROOT) for the given "real" runfolder"""
    assert 'SCRATCH_ROOT' in os.environ, \
    "SCRATCH_ROOT environment variable must be defined"
    try:
        w_2, w_c, E0, pulse_label = runfolder_to_params(runfolder)
        temp_runfolder = os.path.join(os.environ['SCRATCH_ROOT'],
                            "stage1_%s_%d_%d_%d"%(pulse_label, w_2, w_c, E0))
    except ValueError:
        if 'SLURM_JOB_ID' in os.environ:
            temp_runfolder = os.environ['SLURM_JOB_ID']
        else:
            import uuid
            temp_runfolder = str(uuid.uuid4())
    return temp_runfolder


def propagate(runfolder, keep=False):
    """
    Map runfolder -> 2QGate, by propagating or reading from an existing U.dat

    If `keep` is True, keep all files resulting from the propagation in the
    runfolder. Otherwise, only prop.log and U.dat will be kept.
    """
    logger = logging.getLogger(__name__)
    from analytical_pulses import AnalyticalPulse

    gatefile = os.path.join(runfolder, 'U.dat')
    config = os.path.join(runfolder, 'config')
    pulse_json = os.path.join(runfolder, 'pulse.json')
    if not os.path.isfile(gatefile):
        try:
            assert os.path.isfile(config), \
            "No config file in runfolder %s" % runfolder
            assert os.path.isfile(pulse_json), \
            "No pulse.json file in runfolder %s" % runfolder
            temp_runfolder = get_temp_runfolder(runfolder)
            logger.debug("Prepararing temp_runfolder %s", temp_runfolder)
            if os.path.isfile(temp_runfolder):
                # This is simply to clean up after a previous bug
                os.unlink(temp_runfolder)
            QDYN.shutil.mkdir(temp_runfolder)
            shutil.copy(config, temp_runfolder)
            pulse = AnalyticalPulse.read(pulse_json).pulse()
            pulse.write(os.path.join(temp_runfolder, 'pulse.guess'))
            logger.info("Propagating %s", runfolder)
            env = os.environ.copy()
            env['OMP_NUM_THREADS'] = '4'
            start = time.time()
            with open(os.path.join(runfolder, 'prop.log'), 'w', 0) as stdout:
                stdout.write("**** tm_en_gh -- dissipation . \n")
                sp.call(['tm_en_gh', '--dissipation', '.'], cwd=temp_runfolder,
                        stderr=sp.STDOUT, stdout=stdout)
                stdout.write("**** rewrite_dissipation.py. \n")
                sp.call(['rewrite_dissipation.py',], cwd=temp_runfolder,
                        stderr=sp.STDOUT, stdout=stdout)
                stdout.write("**** tm_en_logical_eigenstates.py . \n")
                sp.call(['tm_en_logical_eigenstates.py', '.'],
                        cwd=temp_runfolder, stderr=sp.STDOUT, stdout=stdout)
                stdout.write("**** tm_en_prop . \n")
                sp.call(['tm_en_prop', '.'], cwd=temp_runfolder, env=env,
                        stderr=sp.STDOUT, stdout=stdout)
            shutil.copy(os.path.join(temp_runfolder, 'U.dat'), runfolder)
            end = time.time()
            logger.info("Successfully finished propagating %s (%d seconds)",
                         runfolder, end-start)
        except Exception as e:
            logger.error(e)
        finally:
            if keep:
                sp.call(['rsync', '-a', '%s/'%temp_runfolder, runfolder])
            shutil.rmtree(temp_runfolder)
    else:
        logger.info("Propagating of %s skipped (gatefile already exists)",
                     runfolder)
    U = None
    try:
        U = QDYN.gate2q.Gate2Q(file=gatefile)
    except IOError as e:
        logger.error(e)
    return U


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


def pre_simplex_scan(w_2, w_c):
    """Perform scan for the given qubit and cavity frequency"""
    # create state 1 runfolders and propagate them
    logger = logging.getLogger(__name__)
    logger.info('Running on host %s' % hostname())
    logger.info('*** Generating Runfolders ***')
    runfolders = generate_runfolders(w_2, w_c)
    threadpool_map = make_threadpool_map(get_cpus()/4)
    logger.info('*** Propagate ***')
    threadpool_map(propagate, runfolders)


def main(argv=None):
    """Main routine"""
    from optparse import OptionParser
    if argv is None:
        argv = sys.argv
    arg_parser = OptionParser(
    usage = "%prog [options] <w_2> <w_c>",
    description = __doc__)
    options, args = arg_parser.parse_args(argv)
    try:
        w_2 = float(args[1])
        w_c = float(args[2])
    except IndexError:
        arg_parser.error("w_2 and w_c must be given")
    except ValueError:
        arg_parser.error("w_1 and w_c must be given as floats")
    assert 'SCRATCH_ROOT' in os.environ, \
    "SCRATCH_ROOT environment variable must be defined"
    pre_simplex_scan(w_2, w_c)

if __name__ == "__main__":
    sys.exit(main())
