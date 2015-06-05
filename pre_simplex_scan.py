#!/usr/bin/env python
"""Scan over guess pulses for system with right qubit frequency w_2, and cavity
frequency w_c, given in GHz"""

import sys
import os
import subprocess as sp
import numpy as np
import QDYN
from QDYN.pulse import Pulse, pulse_tgrid, blackman, carrier
from textwrap import dedent
from random import random


def get_cpus():
    """Return number of available cores, either SLURM-assigned cores or number
    of cores on the machine"""
    if 'SLURM_JOB_CPUS_PER_NODE' in os.environ:
        return int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    else:
        sockets = int(sp.check_output(
                  'grep "physical id" /proc/cpuinfo | uniq | wc -l',
                  shell=True))
        cores_per_socket = int(sp.check_output(
                  "grep 'cpu cores' /proc/cpuinfo | uniq | awk '{print $4}'",
                  shell=True))
        return sockets * cores_per_socket


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


def write_run(config, params, pulse, runfolder):
    """Write config file and pulse to runfolder"""
    QDYN.shutil.mkdir(runfolder)
    with open(os.path.join(runfolder, 'config'), 'w') as config_fh:
        config_fh.write(config.format(**params))
    pulse.write(filename=os.path.join(runfolder, 'pulse.guess'))


def generate_jobs(w2, wc):
    """Generate an array of tuples (config, pulse, runfolder)"""
    jobs = []
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
    '''.format(T='{T}', nt='{nt}', w_c=wc*1000.0, w_2=w2*1000))
    T = 200.0
    nt = 200*11*100
    tgrid = pulse_tgrid(200, nt=nt)
    w_min = 5.5 # GHz  -- minimal frequency to allow for pulse

    runfolder_root = 'w2_%dMHz_wc_%dMHz/stage1' % (w2*1000, wc*1000)

    # field-free
    runfolder = os.path.join(runfolder_root, 'field_free')
    pulse = Pulse(tgrid=tgrid, time_unit='ns', ampl_unit='MHz')
    pulse.preamble = ['# Guess pulse: field-free']
    pulse.amplitude = 0.0 * carrier(tgrid, 'ns', 0.0, 'GHz')
    write_run(config, {'T':pulse.T, 'nt':nt}, pulse, runfolder)
    jobs.append(runfolder)

    # single-frequency (center)
    for E0 in [10, 50, 100, 150, 200, 250, 300, 350, 400, 450]:
        runfolder = os.path.join(runfolder_root, '1freq_center', "E%03d"%E0)
        pulse = Pulse(tgrid=tgrid, time_unit='ns', ampl_unit='MHz')
        pulse.preamble = [
        '# Guess pulse: single-frequency (centered), E0 = %d MHz'%E0]
        w_L = 0.5*(6.0 + w2)
        pulse.amplitude = E0 * blackman(tgrid, 0, T) \
                             * carrier(tgrid, 'ns', w_L, 'GHz')
        write_run(config, {'T':pulse.T, 'nt':nt}, pulse, runfolder)
        jobs.append(runfolder)

    # single-frequency (random)
    for realization in xrange(10):
        w_L = w_min + random()*(1.1*wc-w_min)
        for E0 in [10, 50, 100, 150, 200, 250, 300, 350, 400, 450]:
            runfolder = os.path.join(runfolder_root,
                                     '1freq_%d'%(realization+1), "E%03d"%E0)
            pulse = Pulse(tgrid=tgrid, time_unit='ns', ampl_unit='MHz')
            pulse.preamble = [('# Guess pulse: single-frequency w_L = %f GHz, '
            'E0 = %d MHz')%(w_L,E0)]
            pulse.amplitude = E0 * blackman(tgrid, 0, T) \
                                * carrier(tgrid, 'ns', w_L, 'GHz')
            write_run(config, {'T':pulse.T, 'nt':nt}, pulse, runfolder)
            jobs.append(runfolder)

    # two-frequency (resonant)
    for E0 in [10, 50, 100, 150, 200, 250, 300, 350, 400, 450]:
        runfolder = os.path.join(runfolder_root, '2freq_resonant', "E%03d"%E0)
        pulse = Pulse(tgrid=tgrid, time_unit='ns', ampl_unit='MHz')
        pulse.preamble = [
        '# Guess pulse: two-frequency (resonant), E0 = %d MHz'%E0]
        pulse.amplitude = E0 * blackman(tgrid, 0, T) \
                             * carrier(tgrid, 'ns', [6.0, w2], 'GHz')
        write_run(config, {'T':pulse.T, 'nt':nt}, pulse, runfolder)
        jobs.append(runfolder)

    # two-frequency (random)
    for realization in xrange(10):
        w_1 = w_min + random()*(1.1*wc-w_min)
        w_2 = w_min + random()*(1.1*wc-w_min)
        phi = 2*random()
        a_1 = random()
        a_2 = random()
        for E0 in [10, 50, 100, 150, 200, 250, 300, 350, 400, 450]:
            runfolder = os.path.join(runfolder_root,
                                     '2freq_%d'%(realization+1), "E%03d"%E0)
            pulse = Pulse(tgrid=tgrid, time_unit='ns', ampl_unit='MHz')
            pulse.preamble = [('# Guess pulse: freqs = %s GHz, '
            'weights = %s, phase = %f pi, E0 = %d MHz')%(
            str((w_1,w_2)), str((a_1,a_2)), phi, E0)]
            pulse.amplitude = E0 * blackman(tgrid, 0, T) \
                              * carrier(tgrid, 'ns', freq=(w_1, w_2),
                                        freq_unit='GHz', weights=(a_1, a_2),
                                        phases=(0.0, phi))
            write_run(config, {'T':pulse.T, 'nt':nt}, pulse, runfolder)
            jobs.append(runfolder)

    # five-frequency (random)
    for realization in xrange(10):
        freq = w_min + np.random.rand(5) * (1.1*wc-w_min)
        a = np.random.rand(5) - 0.5
        b = np.random.rand(5) - 0.5
        norm_carrier = CRAB_carrier(tgrid, 'ns', freq, 'GHz', a, b,
                                    normalize=True)
        for E0 in [10, 50, 100, 150, 200, 250, 300, 350, 400, 450]:
            runfolder = os.path.join(runfolder_root,
                                     '5freq_%d'%(realization+1), "E%03d"%E0)
            pulse = Pulse(tgrid=tgrid, time_unit='ns', ampl_unit='MHz')
            pulse.preamble = [('# Guess pulse: CRAB (norm.) freqs = %s GHz, '
            'a_n = %s, b_n = %s, E0 = %d MHz')%(str(freq), str(a), str(b), E0)]
            pulse.amplitude = E0 * blackman(tgrid, 0, T) * norm_carrier
            write_run(config, {'T':pulse.T, 'nt':nt}, pulse, runfolder)
            jobs.append(runfolder)

    return jobs


def propagate(runfolder):
    """
    Map runfolder -> 2QGate, by propagating or reading from an existing U.dat
    """
    gatefile = os.path.join(runfolder, 'U.dat')
    if not os.path.isfile(gatefile):
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = '4'
        sp.call(['tm_en_gh', '--dissipation', '.'], cwd=runfolder)
        sp.call(['rewrite_dissipation.py',], cwd=runfolder)
        sp.call(['tm_en_logical_eigenstates.py', '.'], cwd=runfolder)
        with open(os.path.join(runfolder, 'prop.log'), 'w') as stdout:
            sp.call(['tm_en_prop.py', '.'], cwd=runfolder, env=env,
                    stderr=sp.STDOUT, stdout=stdout)
        # TODO: clean up
    return QDYN.gate2q.Gate2Q(file=gatefile)


def threadpool_map(worker, jobs):
    """map worker routine over array of jobs, using a thread pool"""
    from multiprocessing.dummy import Pool
    p = get_cpus() / 4
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


def pre_simplex_scan(w_2, w_c):
    """Perform scan for the given qubit and cavity frequency"""
    # create state 1 runfolders and propagate them
    runfolders =  generate_jobs(w_2, w_c)
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
    pre_simplex_scan(w_2, w_c)

if __name__ == "__main__":
    sys.exit(main())
