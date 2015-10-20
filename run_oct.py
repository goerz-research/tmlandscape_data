#!/usr/bin/env python
"""Run an Optimiztion on the given runfolder"""

import time
import sys
import os
import re
import subprocess as sp
import QDYN
import logging
import numpy as np
from glob import glob
from clusterjob.utils import read_file
from stage2_simplex import get_temp_runfolder
from QDYN.pulse import Pulse


def reset_pulse(pulse, iter):
    """Reset pulse at the given iteration to the last available snapshot,
    assuming that snapshots are available using the same name as pulse, with
    the iter append to the file name (e.g. iter.dat.100 for iter.dat).
    Snapshots must be at least 10 iterations older than the current pulse"""
    snapshot_list = glob("%s.*"%pulse)
    snapshots = {}
    logger = logging.getLogger(__name__)
    logger.debug("resetting in iter %d", iter)
    logger.debug("available snapshots: %s", str(snapshot_list))
    for snapshot in snapshot_list:
        try:
            snapshot_iter =  int(os.path.splitext(snapshot)[1][1:])
            snapshots[snapshot_iter] = snapshot
        except ValueError:
            pass # ignore pulse.dat.prev
    snapshot_iters = sorted(snapshots.keys())
    os.unlink(pulse)
    while len(snapshot_iters) > 0:
        snapshot_iter = snapshot_iters.pop()
        if (iter == 0) or (snapshot_iter + 10 < iter):
            logger.debug("accepted snapshot: %s (iter %d)",
                         snapshots[snapshot_iter], snapshot_iter)
            QDYN.shutil.copy(snapshots[snapshot_iter], pulse)
            return
        else:
            logger.debug("rejected snapshot: %s (iter %d)",
                         snapshots[snapshot_iter], snapshot_iter)
    logger.debug("no accepted snapshot")


def run_oct(runfolder, rwa=False, continue_oct=False):
    """Run optimal control on the given runfolder. Adjust lambda_a if
    necessary.

    Assumes that the runfolder cotnains the files config and pulse.guess, and
    optionally target_gate.dat, pulse.dat, and oct_iters.dat.

    Creates (overwrites) the files pulse.dat and oct_iters.dat.

    Also, a file config.oct is created that contains the last update to
    lambda_a. The original config file will remain unchanged.
    """
    logger = logging.getLogger(__name__)
    temp_runfolder = get_temp_runfolder(runfolder)
    QDYN.shutil.mkdir(temp_runfolder)
    files_to_copy = ['config', 'pulse.guess', 'target_gate.dat']
    if continue_oct:
        files_to_copy.extend(['pulse.dat', 'oct_iters.dat'])
    for file in files_to_copy:
        if os.path.isfile(os.path.join(runfolder, file)):
            QDYN.shutil.copy(os.path.join(runfolder, file), temp_runfolder)
            logger.debug("%s to temp_runfolder %s", file, temp_runfolder)
        else:
            if file in ['config', 'pulse.guess']:
                raise IOError("%s does not exist in %s" % (file, runfolder))
    temp_config = os.path.join(temp_runfolder, 'config')
    temp_pulse_dat = os.path.join(temp_runfolder, 'pulse.dat')
    logger.info("Starting optimization of %s (in %s)", runfolder,
                temp_runfolder)
    with open(os.path.join(runfolder, 'oct.log'), 'w', 0) as stdout:
        cmds = []
        if (rwa):
            cmds.append(['tm_en_gh', '--rwa', '--dissipation', '.'])
        else:
            cmds.append(['tm_en_gh', '--dissipation', '.'])
        cmds.append(['rewrite_dissipation.py',])
        cmds.append(['tm_en_logical_eigenstates.py', '.'])
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = '1'
        for cmd in cmds:
            stdout.write("**** " + " ".join(cmd) +"\n")
            sp.call(cmd , cwd=temp_runfolder, env=env,
                    stderr=sp.STDOUT, stdout=stdout)
        # we assume that the value for lambda_a is badly chosen and iterate
        # over optimizations until we find a good value
        bad_lambda = True
        pulse_explosion = False
        while bad_lambda:
            oct_proc = sp.Popen(['tm_en_oct', '.'], cwd=temp_runfolder,
                                env=env, stdout=sp.PIPE)
            iter = 0
            g_a_int = 0.0
            while True: # monitor STDOUT from oct
                line = oct_proc.stdout.readline()
                if line != '':
                    stdout.write(line)
                    m = re.search(r'^\s*(\d+) \| [\d.E+-]+ \| ([\d.E+-]+) \|',
                                  line)
                    if m:
                        iter = int(m.group(1))
                        g_a_int = float(m.group(2))
                    # Every 50 iterations, we take a snapshot of the current
                    # pulse, so that "bad lambda" restarts continue from there
                    if (iter > 0) and (iter % 50 == 0):
                        QDYN.shutil.copy(temp_pulse_dat,
                                         temp_pulse_dat+'.'+str(iter))
                    # if the pulse changes in first iteration are too small, we
                    # lower lambda_a, unless lambda_a was previously adjusted
                    # to avoid exploding pulse values
                    if iter == 1 and g_a_int < 1.0e-5 and not pulse_explosion:
                        logger.debug("pulse update too small")
                        logger.debug("Kill %d" % oct_proc.pid)
                        oct_proc.kill()
                        scale_lambda_a(temp_config, 0.5)
                        reset_pulse(temp_pulse_dat, iter)
                        break # next bad_lambda loop
                    # if the pulse update explodes, we increase lambda_a (and
                    # prevent it from decreasing again)
                    if ( ('amplitude exceeds maximum value' in line)
                    or   ('Loss of monotonic convergence' in line)
                    or   (g_a_int > 1.0e-1)):
                        pulse_explosion = True
                        if "Loss of monotonic convergence" in line:
                            logger.debug("loss of monotonic conversion")
                        else:
                            logger.debug("pulse explosion")
                        logger.debug("Kill %d" % oct_proc.pid)
                        oct_proc.kill()
                        scale_lambda_a(temp_config, 1.25)
                        reset_pulse(temp_pulse_dat, iter)
                        break # next bad_lambda loop
                    # if there are no significant pulse changes anymore, we
                    # stop the optimization prematurely
                    if iter > 10 and g_a_int < 1.0e-7:
                        logger.debug("pulse update insignificant (converged)")
                        logger.debug("Kill %d" % oct_proc.pid)
                        oct_proc.kill()
                        bad_lambda = False
                        break # effectively break from bad_lambda loop
                else: # line == ''
                    # OCT finished
                    bad_lambda = False
                    break # effectively break from bad_lambda loop
    for file in ['pulse.dat', 'oct_iters.dat']:
        if os.path.isfile(os.path.join(temp_runfolder, file)):
            QDYN.shutil.copy(os.path.join(temp_runfolder, file), runfolder)
    if os.path.isfile(os.path.join(temp_runfolder, 'config')):
        QDYN.shutil.copy(os.path.join(temp_runfolder, 'config'),
                         os.path.join(runfolder, 'config.oct'))
    QDYN.shutil.rmtree(temp_runfolder)
    logger.debug("Removed temp_runfolder %s", temp_runfolder)
    logger.info("Finished optimization")


def scale_lambda_a(config, factor):
    """Scale lambda_a in the given config file with the given factor"""
    QDYN.shutil.copy(config, '%s~'%config)
    logger = logging.getLogger(__name__)
    lambda_a_pt = r'oct_lambda_a\s*=\s*([\deE.+-]+)'
    with open('%s~'%config) as in_fh, open(config, 'w') as out_fh:
        lambda_a = None
        for line in in_fh:
            m = re.search(lambda_a_pt, line)
            if m:
                lambda_a = float(m.group(1))
                lambda_a_new = lambda_a * factor
                logger.info("%s: lambda_a: %.2e -> %.2e"
                            % (config, lambda_a, lambda_a_new))
                line = re.sub(lambda_a_pt,
                              'oct_lambda_a = %.2e'%(lambda_a_new), line)
            out_fh.write(line)
        if lambda_a is None:
            raise ValueError("no lambda_a in %s" % config)


def propagate(runfolder, rwa, keep=False):
    """
    Map runfolder -> 2QGate, by propagating or reading from an existing U.dat

    If `keep` is True, keep all files resulting from the propagation in the
    runfolder. Otherwise, only prop.log and U.dat will be kept.

    Assumes the runfolder contains a file pulse.dat or pulse.guess with the
    pulse to be propagated (pulse.guess is only used if no pulse.dat exists).
    The folder must also contain a matching config file.
    """
    logger = logging.getLogger(__name__)

    gatefile = os.path.join(runfolder, 'U.dat')
    config = os.path.join(runfolder, 'config')
    if re.search('prop_guess\s*=\s*T', read_file(config)):
        raise ValueError("prop_guess must be set to F in %s" % config)
    pulse_dat = os.path.join(runfolder, 'pulse.dat')
    pulse_guess = os.path.join(runfolder, 'pulse.guess')
    target_gate_dat = os.path.join(runfolder, 'target_gate.dat')
    if not os.path.isfile(gatefile):
        try:
            assert os.path.isfile(config), \
            "No config file in runfolder %s" % runfolder
            temp_runfolder = get_temp_runfolder(runfolder)
            logger.debug("Prepararing temp_runfolder %s", temp_runfolder)
            QDYN.shutil.mkdir(temp_runfolder)
            QDYN.shutil.copy(pulse_guess, temp_runfolder)
            if os.path.isfile(target_gate_dat):
                QDYN.shutil.copy(target_gate_dat, temp_runfolder)
            if os.path.isfile(pulse_dat):
                QDYN.shutil.copy(pulse_dat, temp_runfolder)
            else:
                QDYN.shutil.copy(pulse_guess,
                                 os.path.join(temp_runfolder, 'pulse.dat'))
            QDYN.shutil.copy(config, temp_runfolder)
            logger.info("Propagating %s", runfolder)
            env = os.environ.copy()
            env['OMP_NUM_THREADS'] = '1'
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
            QDYN.shutil.copy(os.path.join(temp_runfolder, 'U.dat'), runfolder)
            end = time.time()
            logger.info("Finished propagating %s (%d seconds)",
                         runfolder, end-start)
        except Exception as e:
            logger.error(e)
        finally:
            if keep:
                sp.call(['rsync', '-a', '%s/'%temp_runfolder, runfolder])
            QDYN.shutil.rmtree(temp_runfolder)
    else:
        logger.info("Propagating of %s skipped (gatefile already exists)",
                     runfolder)
    U = None
    try:
        U = QDYN.gate2q.Gate2Q(file=gatefile)
        if np.isnan(U).any():
            logger.error("gate %s contains NaN", gatefile)
    except IOError as e:
        logger.error(e)
    return U


def get_iter_stop(config):
    """Extract the value of iter_stop from the given config file"""
    with open(config) as in_fh:
        for line in in_fh:
            m = re.search(r'iter_stop\s*=\s*(\d+)', line)
            if m:
                return int(m.group(1))
    return None


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
    arg_parser.add_option(
        '--continue', action='store_true', dest='cont',
        default=False, help="Continue from an existing pulse.dat")
    arg_parser.add_option(
        '--debug', action='store_true', dest='debug',
        default=False, help="Enable debugging output")
    options, args = arg_parser.parse_args(argv)
    try:
        runfolder = args[1]
        if not os.path.isdir(runfolder):
            arg_parser.error("runfolder %s does not exist"%runfolder)
    except IndexError:
        arg_parser.error("runfolder be given")
    assert 'SCRATCH_ROOT' in os.environ, \
    "SCRATCH_ROOT environment variable must be defined"
    iter_stop = get_iter_stop(os.path.join(runfolder, 'config'))
    pulse_file = (os.path.join(runfolder, 'pulse.dat'))
    logger = logging.getLogger()
    if options.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    perform_optimization = True
    if os.path.isfile(pulse_file):
        pulse = Pulse(pulse_file)
        if pulse.oct_iter <= 1:
            os.unlink(pulse_file)
            logger.debug("pulse.dat in %s removed as invalid", runfolder)
        else:
            if pulse.oct_iter == iter_stop:
                logger.info("OCT for %s already complete", runfolder)
                perform_optimization = False
    if perform_optimization:
        if os.path.isfile(os.path.join(runfolder, 'U.dat')):
            # if we're doing a new oct, we should delete U.dat
            os.unlink(os.path.join(runfolder, 'U.dat'))
            if os.path.isfile(os.path.join(runfolder, 'U_closest_PE.dat')):
                os.unlink(os.path.join(runfolder, 'U_closest_PE.dat'))
        run_oct(runfolder, rwa=options.rwa, continue_oct=options.cont)
    if not os.path.isfile(os.path.join(runfolder, 'U.dat')):
        propagate(runfolder, rwa=options.rwa)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
