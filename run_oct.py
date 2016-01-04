#!/usr/bin/env python
"""Run an Krotov-optimiztion , and propagate the optimized pulse, on the given
runfolder.

Runfolder must contain pulse.guess, config, and target_gate.dat.
After Krotov optimization, runfolder will contain the additional files
oct.log (log file), pulse.dat (optimized pulse), oct_iters.dat (OCT iteration
data), config.oct (copy of config with automatically determined value of
lambda_a). After propagation, the files U.dat (result of propagating
pulse.dat), and prop.log (log file) will also exist.

No optimization is done if pulse.dat already exists and either
(a) the oct iter in the header matches the "iter_stop" in the config, or
(b) the header contains the word "converged"

If optimization is performed, it starts from the guess pulse. Continuation from
an existing pulse.dat happens only if the --continue option is given.

Using the --pre-simplex option, a simplex optimization of pulse.guess may be
performed. This overwrites pulse.guess with an optimized guess pulse, and
creates the files pulse.json (analytic approximation of the original
pulse.guess), pulse_opt.json (analytic simplex-optimized pulse), simplex.log
(log file), and pulse.guess.pre_simplex (original pulse.guess file)
"""

import time
import sys
import os
import re
import subprocess as sp
import QDYN
import logging
import numpy as np
from numpy.random import random
from glob import glob
from clusterjob.utils import read_file
from stage2_simplex import get_temp_runfolder, run_simplex
from QDYN.pulse import Pulse
from analytical_pulses import AnalyticalPulse
from notebook_utils import get_w_d_from_config, read_target_gate
from clusterjob.utils import read_file

MAX_TRIALS = 200


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


def run_pre_krotov_simplex(runfolder, formula, rwa=False, randomize=False):
    """Read a numerical guess pulse from pulse.guess and map it as closely as
    possible to an analytical pulse, which is stored in pulse.json. Then, a
    simplex optimization is performed to optimize towards a target gate that
    must be defined in target_gate.dat. The result of this simplex optimization
    is stored in pulse_opt.json. the original file pulse.guess is copied to
    pulse.guess.pre_simplex, and the pulse described in pulse_opt.json is
    written out as a numerical pulse to pulse.guess.

    If pulse_opt.json already exists, no simplex optimization is performed.
    """
    for file in ['config', 'pulse.guess', 'target_gate.dat']:
        file = os.path.join(runfolder, file)
        if not os.path.isfile(file):
            raise IOError("%s does not exist" % file)
    logger = logging.getLogger(__name__)
    pulse_opt_json = os.path.join(runfolder, 'pulse_opt.json')
    pulse_guess_pre_simplex = os.path.join(runfolder, 'pulse.guess.pre_simplex')
    pulse_guess = os.path.join(runfolder, 'pulse.guess')
    p_guess = Pulse(filename=pulse_guess)
    if os.path.isfile(pulse_opt_json):
        p_analytic = AnalyticalPulse.read(pulse_opt_json)
        # Check that pulse.guess matches pulse_opt.json
        p_guess_matches_p_analytic = False
        for line in p_guess.preamble:
            if p_analytic.header in line:
                p_guess_matches_p_analytic = True
        if not p_guess_matches_p_analytic:
            if os.path.isfile(pulse_guess_pre_simplex):
                os.unlink(pulse_guess)
                # below, we'll copy pulse.guess.pre_simplex to pulse.guess
            else:
                raise IOError("Something is seriously wrong with the "
                                "guess pulse")
        if p_analytic.formula_name == formula:
            logger.debug("Skipping simplex because pulse_opt.json "
                         "already exists")
            return # skip simplex
        else:
            logger.debug("pulse_opt.json exists, but is using the "
                         "wrong formula")
            os.unlink(pulse_opt_json)
            if os.path.isfile(pulse_guess_pre_simplex):
                os.unlink(pulse_guess)
                # below, we'll copy pulse.guess.pre_simplex to pulse.guess
            else:
                raise IOError("Something is seriously wrong with the "
                                "guess pulse")
    # if pulse.guess.pre_simplex is around, we revert to that
    if os.path.isfile(pulse_guess_pre_simplex):
            logger.debug("Reverting to original pulse.guess")
            QDYN.shutil.copy(pulse_guess_pre_simplex, pulse_guess)
            os.unlink(pulse_guess_pre_simplex)

    # if we're going to do a new pre-simplex optimization, we have to delete
    # any files that will be generated by Krotov, as they are now invalid
    # (starting from the wrong guess pulse)
    for file in ['oct.log', 'pulse.dat', 'oct_iters.dat', 'config.oct',
            'U.dat', 'prop.log', 'simplex.log']:
        file = os.path.join(runfolder, file)
        if os.path.isfile(file):
            os.unlink(file)

    # determine analytical guess pulse (pulse.guess -> pulse.json)
    config = os.path.join(runfolder, 'config')
    if formula == '1freq_rwa':
        assert(rwa)
        w_d = get_w_d_from_config(config)
        E0 = np.max(np.abs(p_guess.amplitude))
        if randomize:
            w_d  += 0.2 * (random() - 1.0) * w_d # 10% variation
            E0   += 0.2 * (random() - 1.0) * E0 # 10% variation
        parameters = {'E0': E0, 'T': p_guess.T, 'w_L': w_d, 'w_d': w_d}
        vary = ['E0', ]; bounds = {'E0': (0.5*E0, 1.5*E0)}
        scipy_options = None
    elif formula == 'CRAB_rwa':
        assert(rwa)
        w_d = get_w_d_from_config(config)
        E0 = np.max(np.abs(p_guess.amplitude))
        if randomize:
            w_d  += 0.2 * (random() - 1.0) * w_d # 10% variation
            E0   += 0.2 * (random() - 1.0) * E0 # 10% variation
            parameters = {'E0': E0, 'T': p_guess.T, 'w_d': w_d,
                          'r': (random(5)-0.5), 'a': random(5),
                          'b': random(5)}
        else:
            parameters = {'E0': E0, 'T': p_guess.T, 'w_d': w_d,
                          'r': np.zeros(5), 'a': np.zeros(5),
                          'b': np.zeros(5)}
        vary = ['E0', 'r', 'a', 'b'];
        bounds = {'E0': (0.8*E0, 1.2*E0), 'r': (-0.5, 0.5), 'a': (0, 1),
                  'b': (0, 1)}
        scipy_options ={'maxfev': 50000}
    else:
        raise ValueError("Don't know what to do with formula %s" % formula)
    if randomize:
        nt = len(p_guess.amplitude) + 1
        guess_analytical = AnalyticalPulse(formula, p_guess.T, nt, parameters,
                t0=0.0, time_unit=p_guess.time_unit,
                ampl_unit=p_guess.ampl_unit, freq_unit=p_guess.freq_unit,
                mode=p_guess.mode)
    else:
        try:
            guess_analytical = AnalyticalPulse.create_from_fit(p_guess,
                                formula=formula, parameters=parameters,
                                vary=vary, bounds=bounds, method='curve_fit')
        except RuntimeError:
            # curve-fit may violate the bounds (-> RuntimeError), in which case
            # L-BFGS-B will (hopefully) find a solution that honors them
            guess_analytical = AnalyticalPulse.create_from_fit(p_guess,
                                formula=formula, parameters=parameters,
                                vary=vary, bounds=bounds, method='L-BFGS-B')
    guess_analytical.write(os.path.join(runfolder, 'pulse.json'))
    logger.debug("Mapped pulse.guess to analytic pulse.json: %s"
                % guess_analytical.header)

    # run simplex optimization (pulse.json -> pulse_opt.json)
    target_gate_dat = os.path.join(runfolder, 'target_gate.dat')
    target_gate = read_target_gate(target_gate_dat)
    assert re.search('prop_guess\s*=\s*F', read_file(config))
    assert re.search('oct_outfile\s*=\s*pulse.dat', read_file(config))
    logger.debug("Running simplex to optimize for target gate")
    run_simplex(runfolder, target=target_gate, rwa=rwa,
                prop_pulse_dat='pulse.dat',
                extra_files_to_copy=['target_gate.dat', ],
                options=scipy_options)

    # switch pulse.guess from original to simplex-optimized version
    p_guess.write(pulse_guess_pre_simplex)
    p_guess = AnalyticalPulse.read(pulse_opt_json).pulse()
    logger.debug("Storing simplex result in pulse.guess (keep original in "
                 "pulse.guess.pre_simplex")
    p_guess.write(os.path.join(runfolder, "pulse.guess"))


def run_oct(runfolder, rwa=False, continue_oct=False):
    """Run optimal control on the given runfolder. Adjust lambda_a if
    necessary.

    Assumes that the runfolder contains the files config and pulse.guess, and
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
        trial = 0
        given_up = False
        while bad_lambda:
            trial += 1
            if trial > MAX_TRIALS:
                bad_lambda = False
                given_up = True
                break # give up
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
                        try:
                            g_a_int = float(m.group(2))
                        except ValueError:
                            # account for Fortran dropping the 'E' in negative
                            # 3-digit exponents
                            g_a_int = float(m.group(2).replace('-','E-'))
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
                        # add a comment to pulse.dat to mark it converged
                        p = Pulse(filename=temp_pulse_dat)
                        p.preamble.append("# converged")
                        p.write(temp_pulse_dat)
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
    if given_up:
        # Giving up is permanent, so we can mark the guess pulse as final
        # by storing it as the optimized pulse. That should prevent pointlessly
        # re-runing OCT
        if not os.path.isfile(os.path.join(runfolder, 'pulse.dat')):
            QDYN.shutil.copy(os.path.join(runfolder, 'pulse.guess'),
                             os.path.join(runfolder, 'pulse.dat'))
        logger.info("Finished optimization (given up after too many "
                    "attempts): %s" % runfolder)
    else:
        logger.info("Finished optimization: %s" % runfolder)


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
                logger.debug("%s: lambda_a: %.2e -> %.2e"
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
    for file in ['config', 'pulse.guess', 'pulse.dat', 'target_gate.dat']:
        file = os.path.join(runfolder, file)
        if not os.path.isfile(file):
            raise IOError("%s does not exist" % file)
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
    arg_parser.add_option(
        '--prop-only', action='store_true', dest='prop_only',
        default=False, help="Only propagate, instead of doing OCT")
    arg_parser.add_option(
        '--keep', action='store_true', dest='keep',
        default=False, help="Keep all files from the propagation")
    arg_parser.add_option(
        '--pre-simplex', action='store', dest='formula',
        help="Run simplex pre-optimization before Krotov")
    arg_parser.add_option(
        '--randomize', action='store_true', dest='randomize',
        default=False, help="In combination with --pre-simplex, start from "
        "a randomized analytic guess pulse, instead of one matching the "
        "original numerical pulse.guess as closely as possible.")
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
    if options.prop_only:
        perform_optimization = False
    else:
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
            for line in pulse.preamble:
                if "converged" in line:
                    perform_optimization = False
    if perform_optimization:
        if options.formula is not None:
            run_pre_krotov_simplex(runfolder, formula=options.formula,
                                   rwa=options.rwa,
                                   randomize=options.randomize)
        if os.path.isfile(os.path.join(runfolder, 'U.dat')):
            # if we're doing a new oct, we should delete U.dat
            os.unlink(os.path.join(runfolder, 'U.dat'))
            if os.path.isfile(os.path.join(runfolder, 'U_closest_PE.dat')):
                os.unlink(os.path.join(runfolder, 'U_closest_PE.dat'))
        run_oct(runfolder, rwa=options.rwa, continue_oct=options.cont)
    if not os.path.isfile(os.path.join(runfolder, 'U.dat')):
        propagate(runfolder, rwa=options.rwa, keep=options.keep)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
