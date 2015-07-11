#!/usr/bin/env python
"""Scan over guess pulses for system with right qubit frequency w_2, and cavity
frequency w_c, given in GHz"""

import sys
import os
import subprocess as sp
import numpy as np
import multiprocessing
import shutil
from textwrap import dedent
from random import random
from functools import partial
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


def make_random_freq(w_1, w_2, w_c, alpha_1, alpha_2, sidebands=True):
    """Return a random_freq function for generating random frequencies in the
    region of the system-typical frequencies.

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
    def random_freq(n):
        """Return n random frequencies"""
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
        return np.array(result)
    return random_freq


def generate_runfolders(runs, w2, wc, T, rwa=False):
    """Generate a set of runfolders, ready for propagation. Returns a list of
    runfolder paths.

    runs: root folder in which to generate runs
    w2:   right qubit frequency [GHz]
    wc:   cavity frequency [GHz]
    T:    gate duration [ns]
    rwa:  if True, write runs in the rotating wave approximation
    """
    from analytical_pulses import AnalyticalPulse
    logger = logging.getLogger(__name__)
    logger.debug("Entering generate_runfolders")
    nt = 200*11*100 # for non-rwa
    config = dedent(r'''
    tgrid: n = 1
    1 : t_start = 0.0, t_stop = {T}_ns, nt = {nt}

    pulse: n = 1
    1: type = file, filename = pulse.guess, id = 1, check_tgrid = F, &
       time_unit = ns, ampl_unit = MHz, is_complex = {is_complex}

    misc: prop = newton, mass = 1.0

    user_ints: n_qubit = 5, n_cavity = 6

    user_strings: gate = CPHASE, J_T = SM

    user_logicals: prop_guess = T, dissipation = T

    user_reals: &
    w_c     = {w_c}_MHz, &
    w_1     = 6000.0_MHz, &
    w_2     = {w_2}_MHz, &
    w_d     = {w_d}_MHz, &
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
    ''')
    runfolders = []
    random_freq = make_random_freq(w_1=6.0, w_2=w2, w_c=wc, alpha_1=0.29,
                                   alpha_2=0.31, sidebands=True)

    runfolder_root = 'runs/w2_%dMHz_wc_%dMHz/stage1' % (w2*1000, wc*1000)

    def runfolder_exists(runfolder):
        return (os.path.isfile(os.path.join(runfolder, 'config'))
            and os.path.isfile(os.path.join(runfolder, 'pulse.json')))

    def write_runfolder(runfolder, analytical_pulse, config, rwa):
        """Write pulse.json and config to given runfolder

        analytical_pulse must be in the lab frame, config must have
        unevaluated placeholders for T, nt, w_c, w_2, w_d, is_complex.

        If rwa is True, the pulse will be transformed in-place to the RWA, and
        an approriate value for nt will be used
        """
        QDYN.shutil.mkdir(runfolder)
        if rwa:
            from notebook_utils import avg_freq, max_freq_delta
            w_d = avg_freq(analytical_pulse) # GHz
            w_max = max_freq_delta(analytical_pulse, w_d) # GHZ
            nt_rwa = int(max(1000, 100 * w_max * analytical_pulse.T))
            # transform pulse
            analytical_pulse.nt = nt_rwa
            if analytical_pulse._formula != 'field_free':
                analytical_pulse._formula += '_rwa'
                analytical_pulse.parameters['w_d'] = w_d
            config_text = config.format(T=T, nt=nt_rwa, w_c=(float(wc)*1000.0),
                                        w_2=(float(w2)*1000.0),
                                        w_d=(float(w_d)*1000.0),
                                        is_complex='T')
        else:
            config_text = config.format(T=T, nt=nt, w_c=(float(wc)*1000.0),
                                        w_2=(float(w2)*1000.0), w_d=0.0,
                                        is_complex='F')
        with open(os.path.join(runfolder, 'config'), 'w') as config_fh:
            config_fh.write(config_text)
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
        write_runfolder(runfolder, pulse, config, rwa)
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
            write_runfolder(runfolder, pulse, config, rwa)
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
                write_runfolder(runfolder, pulse, config, rwa)
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
            write_runfolder(runfolder, pulse, config, rwa)
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
                write_runfolder(runfolder, pulse, config, rwa)
            runfolders.append(runfolder)

    # five-frequency (random)
    for realization in xrange(10):
        freq_high = random_freq(2)
        a_high    = np.random.rand(2) - 0.5
        b_high    = np.random.rand(2) - 0.5
        freq_low = 0.025 * np.random.rand(3) # GHz
        a_low    = np.random.rand(3) - 0.5
        b_low    = np.random.rand(3) - 0.5
        for E0 in amplitudes:
            pulse_label = '5freq_%d'%(realization+1)
            runfolder = os.path.join(runfolder_root, pulse_label, "E%03d"%E0)
            if runfolder_exists(runfolder):
                logger.debug("%s exists (skipped)", runfolder)
            else:
                logger.debug("Generating %s", runfolder)
                pulse = AnalyticalPulse('5freq', T, nt,
                        parameters={'E0': E0, 'T': T, 'freq_high': freq_high,
                                    'a_high': a_high, 'b_high': b_high,
                                    'freq_low': freq_low, 'a_low': a_low,
                                    'b_low': b_low},
                        time_unit='ns', ampl_unit='MHz')
                write_runfolder(runfolder, pulse, config, rwa)
            runfolders.append(runfolder)

    logger.debug("Finished generate_runfolders")
    return runfolders


def get_temp_runfolder(runfolder):
    """Return the path for an appropriate temporary runfolder (inside
    $SCRATCH_ROOT) for the given "real" runfolder"""
    assert 'SCRATCH_ROOT' in os.environ, \
    "SCRATCH_ROOT environment variable must be defined"
    if 'SLURM_JOB_ID' in os.environ:
        temp_runfolder = os.environ['SLURM_JOB_ID']
    else:
        import uuid
        temp_runfolder = str(uuid.uuid4())
    return os.path.join(os.environ['SCRATCH_ROOT'], temp_runfolder)


def propagate(runfolder, rwa, keep=False):
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
            if pulse.formula_name.endswith('_rwa'):
                assert rwa, "RWA pulse must be propagated in RWA"
            pulse.write(os.path.join(temp_runfolder, 'pulse.guess'))
            logger.info("Propagating %s", runfolder)
            env = os.environ.copy()
            env['OMP_NUM_THREADS'] = '4'
            start = time.time()
            with open(os.path.join(runfolder, 'prop.log'), 'w', 0) as stdout:
                cmds = []
                if (rwa):
                    cmds.append(['tm_en_gh', '--rwa', '--dissipation', '.'])
                else:
                    cmds.append(['tm_en_gh', '--dissipation', '.'])
                cmds.append(['rewrite_dissipation.py',])
                cmds.append(['tm_en_logical_eigenstates.py', '.'])
                cmds.append(['tm_en_prop', '.'])
                for cmd in cmds:
                    stdout.write("**** " + " ".join(cmd) +"\n")
                    sp.call(cmd , cwd=temp_runfolder, env=env,
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


def pre_simplex_scan(runs, w2, wc, T, rwa=False):
    """Perform scan for the given qubit and cavity frequency

    runs: root folder in which to generate runs
    w2:   right qubit frequency [GHz]
    wc:   cavity frequency [GHz]
    T:    gate duration [ns]
    rwa:  if True, write runs in the rotating wave approximation
    """
    # create state 1 runfolders and propagate them
    logger = logging.getLogger(__name__)
    logger.info('Running on host %s' % hostname())
    logger.info('*** Generating Runfolders ***')
    runfolders = generate_runfolders(runs, w2, wc, T, rwa=rwa)
    threadpool_map = make_threadpool_map(get_cpus()/4)
    logger.info('*** Propagate ***')
    worker = partial(propagate, rwa=rwa, keep=True) # DEBUG
    threadpool_map(worker, runfolders)


def main(argv=None):
    """Main routine"""
    from optparse import OptionParser
    if argv is None:
        argv = sys.argv
    arg_parser = OptionParser(
    usage = "%prog [options] <runs> <w_2 GHz> <w_c GHz> <T ns>",
    description = __doc__)
    arg_parser.add_option(
        '--rwa', action='store_true', dest='rwa',
        default=False, help="Perform all calculations in the RWA.")
    options, args = arg_parser.parse_args(argv)
    try:
        runs = args[1]
        w2   = float(args[2]) # GHz
        wc   = float(args[3]) # GHz
        T    = float(args[4]) # ns
    except IndexError:
        arg_parser.error("Missing arguments")
    except ValueError:
        arg_parser.error("w1, wc, T must be given as floats")
    assert 'SCRATCH_ROOT' in os.environ, \
    "SCRATCH_ROOT environment variable must be defined"
    pre_simplex_scan(runs, w2, wc, T, rwa=options.rwa)

if __name__ == "__main__":
    sys.exit(main())
