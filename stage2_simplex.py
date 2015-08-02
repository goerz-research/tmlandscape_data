#!/usr/bin/env python
"""Scan over guess pulses for system with right qubit frequency w_2, and cavity
frequency w_c, given in GHz"""

import sys
import os
import subprocess as sp
import numpy as np
import time
import QDYN
import logging
import scipy.optimize
from analytical_pulses import AnalyticalPulse
from notebook_utils import pulse_config_compat, avg_freq, max_freq_delta, \
                           J_target
from QDYNTransmonLib.io import read_params
logging.basicConfig(level=logging.INFO)

def hostname():
    """Return the hostname"""
    import socket
    return socket.gethostname()


def get_temp_runfolder(runfolder):
    """Return the path for an appropriate temporary runfolder (inside
    $SCRATCH_ROOT) for the given "real" runfolder"""
    assert 'SCRATCH_ROOT' in os.environ, \
    "SCRATCH_ROOT environment variable must be defined"
    import uuid
    temp_runfolder = str(uuid.uuid4())
    if 'SLURM_JOB_ID' in os.environ:
        temp_runfolder = "%s_$s" % (os.environ['SLURM_JOB_ID'], temp_runfolder)
    return os.path.join(os.environ['SCRATCH_ROOT'], temp_runfolder)


def pulse_frequencies_ok(analytical_pulse, system_params):
    """Return True if all the frequencies in the analytical pulse are within a
    reasonable interval around the system frequencies, False otherwise"""
    w_1 = system_params['w_1'] # GHZ
    w_2 = system_params['w_2'] # GHZ
    w_c = system_params['w_c'] # GHZ
    alpha_1 = system_params['alpha_1'] # GHZ
    alpha_2 = system_params['alpha_2'] # GHZ
    delta = abs(w_2 - w_1) # GHz
    p = analytical_pulse.parameters
    if analytical_pulse.formula_name == 'field_free':
        return True
    elif analytical_pulse.formula_name in ['1freq', '1freq_rwa']:
        if w_1-1.2*abs(alpha_1) <= p['w_L'] <= w_2+0.2*abs(alpha_2):
            return True
        if w_c-1.2*delta <= p['w_L'] <= w_c+1.2*delta:
            return True
    elif analytical_pulse.formula_name in ['2freq', '2freq_rwa']:
        for param in ['freq_1', 'freq_2']:
            if w_1-1.2*abs(alpha_1) <= p[param] <= w_2+0.2*abs(alpha_2):
                return True
            if w_c-1.2*delta <= p[param] <= w_c+1.2*delta:
                return True
    elif analytical_pulse.formula_name in ['5freq', '5freq_rwa']:
        for w in p['freq_high']:
            if w_1-1.2*abs(alpha_1) <= w <= w_2+0.2*abs(alpha_2):
                return True
            if w_c-1.2*delta <= w <= w_c+1.2*delta:
                return True
    else:
        raise ValueError("Unknown formula name: %s"
                         % analytical_pulse.formula_name)
    return False


def run_simplex(runfolder, target, rwa=False):
    """Run a simplex over all the pulse parameters, optimizing towards the
    given target ('PE' or 'SQ')
    Write the optimized pulse out as pulse.dat
    """

    logger = logging.getLogger(__name__)
    cachefile = os.path.join(runfolder, 'get_U.cache')
    config = os.path.join(runfolder, 'config')
    # We need the system parameters to ensure that the pulse frequencies stay
    # in a reasonable range. They don't change over the course of the simplex,
    # so we can get get them once in the beginning.
    system_params = read_params(config, 'GHz')
    pulse0 = os.path.join(runfolder, 'pulse.json')
    assert os.path.isfile(config), "Runfolder must contain config"
    assert os.path.isfile(pulse0), "Runfolder must contain pulse.json"
    temp_runfolder = get_temp_runfolder(runfolder)
    QDYN.shutil.mkdir(temp_runfolder)
    QDYN.shutil.copy(config, temp_runfolder)
    pulse = AnalyticalPulse.read(pulse0)
    parameters = [p for p in sorted(pulse.parameters.keys())
                  if p not in ['T', 'w_d']]
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '4'

    @QDYN.memoize.memoize
    def get_U(x, pulse):
        """Return the resulting gate for the given pulse. The argument 'x' is
           not used except as a key for memoize
        """
        pulse.pulse().write(os.path.join(temp_runfolder, 'pulse.guess'))
        # rewrite config file to match pulse (time grid and w_d)
        pulse_config_compat(pulse, os.path.join(temp_runfolder, 'config'),
                            adapt_config=True)
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
        U = QDYN.gate2q.Gate2Q(file=os.path.join(temp_runfolder, 'U.dat'))
        return U
    get_U.load(cachefile)

    def f(x, log_fh=None):
        """function to minimize. Modifies 'pulse' from outer scope based on x
        array, then call get_U to obtain figure of merit"""
        pulse.array_to_parameters(x, parameters)
        if ( (not pulse_frequencies_ok(pulse, system_params)) \
        or (abs(pulse.parameters['E0']) > 1500.0) ):
            J = 10.0 # infinitely bad
            logger.info("%s -> %f", pulse.header, J)
            if log_fh is not None:
                log_fh.write("%s -> %f\n" % (pulse.header, J))
                return J
        if rwa:
            w_d = avg_freq(pulse) # GHz
            w_max = max_freq_delta(pulse, w_d) # GHZ
            pulse.parameters['w_d'] = w_d
            pulse.nt = int(max(2000, 100 * w_max * pulse.T))
        U = get_U(x, pulse)
        C = U.closest_unitary().concurrence()
        max_loss = np.max(1.0 - U.logical_pops())
        J = J_target(target, C, max_loss)
        logger.info("%s -> %f", pulse.header, J)
        if log_fh is not None:
            log_fh.write("%s -> %f\n" % (pulse.header, J))
        return J

    def dump_cache(dummy):
        get_U.dump(cachefile)

    x0 = pulse.parameters_to_array(keys=parameters)
    try:
        with open(os.path.join(runfolder, 'simplex.log'), 'a', 0) as log_fh:
            log_fh.write("%s\n" % time.asctime())
            res = scipy.optimize.minimize(f, x0, method='Nelder-Mead',
                  options={'maxfev': 100*len(parameters), 'xtol': 0.1,
                           'ftol': 0.05},
                  args=(log_fh, ), callback=dump_cache)
        pulse.array_to_parameters(res.x, parameters)
        get_U.func(res.x, pulse) # memoization disabled
        QDYN.shutil.copy(os.path.join(temp_runfolder, 'config'), runfolder)
        QDYN.shutil.copy(os.path.join(temp_runfolder, 'U.dat'), runfolder)
        pulse.write(os.path.join(runfolder, 'pulse_opt.json'), pretty=True)
    finally:
        get_U.dump(cachefile)
        QDYN.shutil.rmtree(temp_runfolder)
    logger.info("Finished optimization: %s" % res.message)


def get_target(runfolder):
    """Extract the target from the given runfolder name"""
    parts = runfolder.split(os.path.sep)
    part = parts.pop()
    if part == '':
        part = parts.pop()
    if part.startswith('PE'):
        return 'PE'
    elif part.startswith('SQ'):
        return 'SQ'
    else:
        raise ValueError("Could not extract target from runfolder %s" %
                         runfolder)


def main(argv=None):
    """Main routine"""
    from optparse import OptionParser
    if argv is None:
        argv = sys.argv
    arg_parser = OptionParser(
    usage = "%prog [options] <runfolder>",
    description = __doc__)
    arg_parser.add_option(
        '--rwa', action='store_true', dest='rwa',
        default=False, help="Perform all calculations in the RWA.")
    options, args = arg_parser.parse_args(argv)
    try:
        runfolder = args[1]
        if not os.path.isdir(runfolder):
            arg_parser.error("runfolder %s does not exist"%runfolder)
    except IndexError:
        arg_parser.error("runfolder be given")
    assert 'SCRATCH_ROOT' in os.environ, \
    "SCRATCH_ROOT environment variable must be defined"
    os.path.split
    run_simplex(runfolder, get_target(runfolder), rwa=options.rwa)


if __name__ == "__main__":
    sys.exit(main())
