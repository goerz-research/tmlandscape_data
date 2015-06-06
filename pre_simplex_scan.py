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
from QDYN.pulse import Pulse, pulse_tgrid, blackman, carrier


def get_cpus():
    """Return number of available cores, either SLURM-assigned cores or number
    of cores on the machine"""
    if 'SLURM_JOB_CPUS_PER_NODE' in os.environ:
        return int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    else:
        return multiprocessing.cpu_count()


def CRAB_carrier(t, time_unit, freq, freq_unit, a, b, normalize=False):
    r'''
    Construct a "carrier" based on the CRAB formula

        .. math::
        E(t) = \sum_{n} (a_n \cos(\omega_n t) + b_n \cos(\omega_n t))

    where :math:`a_n` is the n'th element of `a`, :math:`b_n` is the n'th
    element of `b`, and :math:`\omega_n` is the n'th element of freq.

    Parameters
    ----------
    t : array-like
        time grid values
    time_unit : str
        Unit of `t`
    freq : scalar, ndarray(float64)
        Carrier frequency or frequencies
    freq_unit : str
        Unit of `freq`
    a: array-like
        Coefficients for cosines
    b: array-line
        Coefficients for sines
    normalize: logical, optional
        If True, normalize the resulting carrier such that its values are in
        [-1,1]

    Notes
    -----

    `freq_unit` can be Hz (GHz, MHz, etc), describing the frequency directly,
    or any energy unit, in which case the energy value E (given through the
    freq parameter) is converted to an actual frequency as

     .. math:: f = E / (\\hbar * 2 * pi)
    '''
    from QDYN.units import NumericConverter
    convert = NumericConverter()
    c = convert.to_au(1, time_unit) * convert.to_au(1, freq_unit)
    assert len(a) == len(b) == len(freq), \
    "freq, a, b must all be of the same length"
    signal = np.zeros(len(t), dtype=np.complex128)
    for w_n, a_n, b_n in zip(freq, a, b):
        signal += a_n * np.cos(c*w_n*t) + b_n * np.sin(c*w_n*t)
    if normalize:
        signal *= 1.0 / (np.abs(signal).max())
    return signal


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


def nstr(a):
    """Return string without newlines"""
    return str(a).replace("\n", " ")


def write_run(args):
    """Create runfolder, write config file and pulse to runfolder

    args is a tuple (runfolder, temp_runfolder, kwargs) where runfolder is the
    full path to the runfolder to be generated, temp_runfolder is the scratch
    space runfolder to be generated, and kwargs is a dictionary of additional
    parameters
    """
    runfolder, temp_runfolder, kwargs = args
    w_2, w_c, E0, pulse_label = runfolder_to_params(runfolder)
    T = 200.0 # ns
    nt = 200*11*100
    QDYN.shutil.mkdir(runfolder)
    QDYN.shutil.mkdir(temp_runfolder)
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
    J       =   5.0_MHz, &
    g_1     = 100.0_MHz, &
    g_2     = 100.0_MHz, &
    n0_qubit  = 0.0, &
    n0_cavity = 0.0, &
    kappa   = 0.05_MHz, &
    gamma_1 = 0.012_MHz, &
    gamma_2 = 0.012_MHz, &
    '''.format(T=T, nt=nt, w_c=w_c, w_2=w_2))
    w_c *= 1/1000.0 # to GHZ
    w_2 *= 1/1000.0 # to GHZ
    tgrid = pulse_tgrid(T, nt=nt)

    if pulse_label == 'field_free':
        pulse = Pulse(tgrid=tgrid, time_unit='ns', ampl_unit='MHz')
        pulse.preamble = ['# Guess pulse: field-free']
        pulse.amplitude = 0.0 * carrier(tgrid, 'ns', 0.0, 'GHz')
    elif pulse_label == '1freq_center':
        pulse = Pulse(tgrid=tgrid, time_unit='ns', ampl_unit='MHz')
        pulse.preamble = [
        '# Guess pulse: single-frequency (centered), E0 = %d MHz'%E0]
        w_L = 0.5*(6.0 + w_2)
        pulse.amplitude = E0 * blackman(tgrid, 0, T) \
                             * carrier(tgrid, 'ns', w_L, 'GHz')
    elif pulse_label.startswith('1freq_'):
        w_L = kwargs['w_L']
        pulse = Pulse(tgrid=tgrid, time_unit='ns', ampl_unit='MHz')
        pulse.preamble = [('# Guess pulse: single-frequency w_L = %f GHz, '
        'E0 = %d MHz')%(w_L,E0)]
        pulse.amplitude = E0 * blackman(tgrid, 0, T) \
                            * carrier(tgrid, 'ns', w_L, 'GHz')
    elif pulse_label == '2freq_resonant':
        pulse = Pulse(tgrid=tgrid, time_unit='ns', ampl_unit='MHz')
        pulse.preamble = [
        '# Guess pulse: two-frequency (resonant), E0 = %d MHz'%E0]
        pulse.amplitude = E0 * blackman(tgrid, 0, T) \
                             * carrier(tgrid, 'ns', [6.0, w_2], 'GHz')
    elif pulse_label.startswith('2freq_'):
        freq_1 = kwargs['freq_1']
        freq_2 = kwargs['freq_2']
        phi = kwargs['phi']
        a_1 = kwargs['a_1']
        a_2 = kwargs['a_2']
        pulse = Pulse(tgrid=tgrid, time_unit='ns', ampl_unit='MHz')
        pulse.preamble = [('# Guess pulse: freqs = %s GHz, '
        'weights = %s, phase = %f pi, E0 = %d MHz')%(
        str((freq_1,freq_2)), str((a_1,a_2)), phi, E0)]
        pulse.amplitude = E0 * blackman(tgrid, 0, T) \
                            * carrier(tgrid, 'ns', freq=(freq_1, freq_2),
                                    freq_unit='GHz', weights=(a_1, a_2),
                                    phases=(0.0, phi))
    elif pulse_label.startswith('5freq_'):
        freq = kwargs['freq']
        a = kwargs['a']
        b = kwargs['b']
        norm_carrier = CRAB_carrier(tgrid, 'ns', freq, 'GHz', a, b,
                                    normalize=True)
        pulse = Pulse(tgrid=tgrid, time_unit='ns', ampl_unit='MHz')
        pulse.preamble = [('# Guess pulse: CRAB (norm.) freqs = %s GHz, '
        'a_n = %s, b_n = %s, E0 = %d MHz')%(nstr(freq), nstr(a), nstr(b), E0)]
        pulse.amplitude = E0 * blackman(tgrid, 0, T) * norm_carrier
    else:
        raise ValueError("Unknown pulse label %s" % pulse_label)

    with open(os.path.join(runfolder, 'config'), 'w') as config_fh:
        config_fh.write(config)
    with open(os.path.join(runfolder, 'pulse.guess.header'), 'w') as ph_fh:
        ph_fh.write("".join(pulse.preamble) + "\n")
    pulse.write(filename=os.path.join(temp_runfolder, 'pulse.guess'))


def generate_runfolders(w2, wc):
    """Generate a set of runfolders, ready for propagation. Returns a list of
    runfolder paths. w2 and wc must be given in GHz"""
    jobs = [] # jobs to be handled by write_run worker
    runfolders = []
    random_freq = make_random_freq(w_1=6.0, w_2=w2, w_c=wc, alpha_1=0.29,
                                   alpha_2=0.31, low_freq_limit=0.02,
                                   sidebands=True)

    runfolder_root = 'w2_%dMHz_wc_%dMHz/stage1' % (w2*1000, wc*1000)

    # field-free
    pulse_label = 'field_free'
    runfolder = os.path.join(runfolder_root, pulse_label)
    temp_runfolder = os.path.join(os.environ['SCRATCH_ROOT'],
                     "stage1_%s_%d_%d_%d"%(pulse_label, w2, wc, 0.0))
    if not os.path.isdir(runfolder):
        jobs.append((runfolder, temp_runfolder, {}))
    runfolders.append((runfolder, temp_runfolder))

    # single-frequency (center)
    for E0 in [10, 50, 100, 150, 200, 250, 300, 350, 400, 450]:
        pulse_label = '1freq_center'
        runfolder = os.path.join(runfolder_root, pulse_label, "E%03d"%E0)
        temp_runfolder = os.path.join(os.environ['SCRATCH_ROOT'],
                        "stage1_%s_%d_%d_%d"%(pulse_label, w2, wc, E0))
        if not os.path.isdir(runfolder):
            jobs.append((runfolder, temp_runfolder, {}))
        runfolders.append((runfolder, temp_runfolder))

    # single-frequency (random)
    for realization in xrange(10):
        w_L = float(random_freq(1)[0])
        for E0 in [10, 50, 100, 150, 200, 250, 300, 350, 400, 450]:
            pulse_label = '1freq_%d'%(realization+1)
            runfolder = os.path.join(runfolder_root, pulse_label, "E%03d"%E0)
            temp_runfolder = os.path.join(os.environ['SCRATCH_ROOT'],
                            "stage1_%s_%d_%d_%d"%(pulse_label, w2, wc, E0))
            if not os.path.isdir(runfolder):
                jobs.append((runfolder, temp_runfolder, {'w_L':w_L}))
            runfolders.append((runfolder, temp_runfolder))

    # two-frequency (resonant)
    for E0 in [10, 50, 100, 150, 200, 250, 300, 350, 400, 450]:
        pulse_label = '2freq_resonant'
        runfolder = os.path.join(runfolder_root, pulse_label, "E%03d"%E0)
        temp_runfolder = os.path.join(os.environ['SCRATCH_ROOT'],
                        "stage1_%s_%d_%d_%d"%(pulse_label, w2, wc, E0))
        if not os.path.isdir(runfolder):
            jobs.append((runfolder, temp_runfolder, {}))
        runfolders.append((runfolder, temp_runfolder))

    # two-frequency (random)
    for realization in xrange(10):
        freq_1, freq_2 = random_freq(2)
        phi = 2*random()
        a_1 = random()
        a_2 = random()
        for E0 in [10, 50, 100, 150, 200, 250, 300, 350, 400, 450]:
            pulse_label = '2freq_%d'%(realization+1)
            runfolder = os.path.join(runfolder_root, pulse_label, "E%03d"%E0)
            temp_runfolder = os.path.join(os.environ['SCRATCH_ROOT'],
                            "stage1_%s_%d_%d_%d"%(pulse_label, w2, wc, E0))
            if not os.path.isdir(runfolder):
                jobs.append((runfolder, temp_runfolder, {'freq_1': freq_1,
                    'freq_2': freq_2, 'phi': phi, 'a_1': a_1, 'a_2': a_2}))
            runfolders.append((runfolder, temp_runfolder))

    # five-frequency (random)
    for realization in xrange(10):
        freq = random_freq(3, n_low_freq=2)
        a = np.random.rand(5) - 0.5
        b = np.random.rand(5) - 0.5
        for E0 in [10, 50, 100, 150, 200, 250, 300, 350, 400, 450]:
            pulse_label = '5freq_%d'%(realization+1)
            runfolder = os.path.join(runfolder_root, pulse_label, "E%03d"%E0)
            temp_runfolder = os.path.join(os.environ['SCRATCH_ROOT'],
                            "stage1_%s_%d_%d_%d"%(pulse_label, w2, wc, E0))
            if not os.path.isdir(runfolder):
                jobs.append((runfolder, temp_runfolder, {'freq': freq.copy(),
                            'a': a.copy(), 'b': b.copy()}))
            runfolders.append((runfolder, temp_runfolder))

    threadpool_map = make_threadpool_map(get_cpus())
    threadpool_map(write_run, jobs)

    return runfolders


def propagate(runfolder_tuple):
    """
    Map runfolder -> 2QGate, by propagating or reading from an existing U.dat
    """
    runfolder, temp_runfolder = runfolder_tuple
    gatefile = os.path.join(runfolder, 'U.dat')
    if not os.path.isfile(gatefile):
        shutil.copy(os.path.join(runfolder, 'config'), temp_runfolder)
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = '4'
        with open(os.path.join(runfolder, 'prop.log'), 'w', 0) as stdout:
            stdout.write("**** tm_en_gh -- dissipation . \n")
            sp.call(['tm_en_gh', '--dissipation', '.'], cwd=temp_runfolder,
                    stderr=sp.STDOUT, stdout=stdout)
            stdout.write("**** rewrite_dissipation.py. \n")
            sp.call(['rewrite_dissipation.py',], cwd=temp_runfolder,
                    stderr=sp.STDOUT, stdout=stdout)
            stdout.write("**** tm_en_logical_eigenstates.py . \n")
            sp.call(['tm_en_logical_eigenstates.py', '.'], cwd=temp_runfolder,
                    stderr=sp.STDOUT, stdout=stdout)
            stdout.write("**** tm_en_prop . \n")
            sp.call(['tm_en_prop', '.'], cwd=temp_runfolder, env=env,
                    stderr=sp.STDOUT, stdout=stdout)
        if 'field_free' in runfolder:
            sp.call('cp {temp_runfolder_all} {runfolder}'.format(
                   temp_runfolder_all=os.path.join(temp_runfolder, '*'),
                   runfolder=runfolder), shell=True)
        else:
            shutil.copy(os.path.join(temp_runfolder, 'U.dat'), runfolder)
        shutil.rmtree(temp_runfolder)
    return QDYN.gate2q.Gate2Q(file=gatefile)


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
    runfolder_tuples =  generate_runfolders(w_2, w_c)
    threadpool_map = make_threadpool_map(get_cpus()/4)
    threadpool_map(propagate, runfolder_tuples)


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
