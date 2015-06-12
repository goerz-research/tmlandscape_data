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
    if 'SLURM_JOB_ID' in os.environ:
        temp_runfolder = os.environ['SLURM_JOB_ID']
    else:
        import uuid
        temp_runfolder = str(uuid.uuid4())
    return os.path.join(os.environ['SCRATCH_ROOT'], temp_runfolder)


def run_simplex(runfolder, target):
    """Run a simplex over all the pulse parameters, optimizing towards the
    given target ('PE' or 'SQ')
    Write the optimized pulse out as pulse.dat
    """

    logger = logging.getLogger(__name__)
    cachefile = os.path.join(runfolder, 'get_U.cache')
    config = os.path.join(runfolder, 'config')
    pulse0 = os.path.join(runfolder, 'pulse.json')
    assert os.path.isfile(config), "Runfolder must contain config"
    assert os.path.isfile(pulse0), "Runfolder must contain pulse.json"
    temp_runfolder = get_temp_runfolder(runfolder)
    QDYN.shutil.mkdir(temp_runfolder)
    QDYN.shutil.copy(config, temp_runfolder)
    pulse = AnalyticalPulse.read(pulse0)
    parameters = [p for p in sorted(pulse.parameters.keys()) if p != 'T']
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '4'

    @QDYN.memoize.memoize
    def get_U(x, pulse):
        """Return the resulting gate for the given pulse
        """
        pulse.pulse().write(os.path.join(temp_runfolder, 'pulse.guess'))
        with open(os.path.join(temp_runfolder, 'prop.log'), 'w', 0) as stdout:
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
        U = QDYN.gate2q.Gate2Q(file=os.path.join(temp_runfolder, 'U.dat'))
        return U
    get_U.load(cachefile)

    def f(x, log_fh=None):
        """function to minimize"""
        pulse.array_to_parameters(x, parameters)
        U = get_U(x, pulse)
        C = U.closest_unitary().concurrence()
        loss = U.pop_loss()
        if target == 'PE':
            J = 1.0 - C + loss
        elif target == 'SQ':
            J = C + loss
        else:
            raise ValueError("Invalid target: %s" % target)
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
                  options={'maxfev': 100*len(parameters), 'ftol': 0.01},
                  args=(log_fh, ), callback=dump_cache)
        pulse.array_to_parameters(res.x, parameters)
        get_U.func(res.x, pulse) # memoization disabled
        QDYN.shutil.copy(os.path.join(temp_runfolder, 'U.dat'), runfolder)
        pulse.write(os.path.join(runfolder, 'pulse_opt.json'), pretty=True)
        QDYN.shutil.copy(os.path.join(temp_runfolder, 'prop.log'), runfolder)
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
    run_simplex(runfolder, get_target(runfolder))


if __name__ == "__main__":
    sys.exit(main())
