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
import os
import re
import json
import zeta_systematic_variation
import subprocess as sp
import QDYN
import logging
import numpy as np
import click
from numpy.random import random
from glob import glob
from clusterjob.utils import read_file
from stage2_simplex import get_temp_runfolder, run_simplex
from QDYN.pulse import Pulse
from analytical_pulses import AnalyticalPulse
from notebook_utils import (get_w_d_from_config, read_target_gate,
        pulse_config_compat, ensure_ham_files, J_target)

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


def write_oct_config(template, config, target, iter_stop=None, max_megs=None,
        max_hours=None, J_T_conv=None, J_T_re=False, lbfgs=False):
    """Write a new config file based on template, but with an updated OCT and
    user_strings section. The `target` parameter must be 'PE' (implies PE
    functional), 'SQ' (implies LI functional), or the name of a file inside the
    runfolder (usually target_gate.dat) that contains an explicit target gate
    (implies SM functional). The remaining parameters set the corresponding
    options in the OCT section.
    """
    with open(template, 'r') as in_fh, open(config, 'w') as out_fh:
        if target == 'PE':
            method = 'krotov2'
            gate = 'CPHASE'
            J_T = 'PE'
        elif target == 'SQ':
            method = 'krotov2'
            gate = 'unity'
            J_T = 'LI'
        else:
            if lbfgs:
                method = 'lbfgs'
            else:
                method = 'krotovpk'
            gate = target
            J_T = 'SM'
            if J_T_re or lbfgs:
                J_T = 'RE'
        section = ''
        for line in in_fh:
            m = re.match(r'^\s*(?P<section>[a-z_]+)\s*:', line)
            if m:
                section = m.group('section')
            if section == 'oct':
                if iter_stop is not None:
                    line = re.sub(r'iter_stop\s*=\s*\d+',
                                  r'iter_stop = %d'%(iter_stop), line)
                if max_megs is not None:
                    line = re.sub(r'max_megs\s*=\s*\d+',
                                  r'max_megs = %d'%(max_megs), line)
                if max_hours is not None:
                    line = re.sub(r'max_hours\s*=\s*\d+',
                                  r'max_hours = %d'%(max_hours), line)
                if J_T_conv is not None:
                    line = re.sub(r'J_T_conv\s*=\s*[\deE+-]+',
                                  r'J_T_conv = %e'%(J_T_conv), line)
                line = re.sub(r'type\s*=\s*\w+', r'type = %s'%(method), line)
                if "iter_dat" in line and 'params_file' not in line:
                    line = re.sub(r'(iter_dat\s*=\s*[\w.]+,)',
                                  r'\1 params_file = oct_params.dat,', line)
            elif section == 'user_strings':
                line = re.sub(r'gate\s*=\s*[\w.]+', r'gate = '+gate, line)
                line = re.sub(r'J_T\s*=\s*\w+', r'J_T = '+J_T, line)
            out_fh.write(line)


def write_prop_config(template, config, pulse_file, rho=False,
        rho_pop_plot=False, n_qubit=None, n_cavity=None, dissipation=None):
    """Write a new config file based on template, updated to propagate the
    numerical pulse stored in `pulse_file`.
    """
    with open(template, 'r') as in_fh, open(config, 'w') as out_fh:
        section = ''
        for line in in_fh:
            m = re.match(r'^\s*(?P<section>[a-z_]+)\s*:', line)
            if m:
                section = m.group('section')
            if section == 'pulse':
                line = re.sub(r'type\s*=\s*\w+', r'type = file', line)
                line = re.sub(r'filename\s*=\s*[\w.]+',
                              r'filename = %s'%(pulse_file), line)
            elif section == 'user_logicals':
                line = re.sub(r'prop_guess\s*=\s*(T|F)',
                              r'prop_guess = T', line)
            if rho_pop_plot:
                line = re.sub(
                    r'rho_prop_mode\s*=\s*(full|pop_dynamics)',
                    r'rho_prop_mode = pop_dynamics', line)
            else:
                line = re.sub(
                    r'rho_prop_mode\s*=\s*(full|pop_dynamics)',
                    r'rho_prop_mode = full', line)
            if n_qubit is not None:
                line = re.sub(r'n_qubit\s*=\s*\d+',
                            r'n_qubit = %d' % n_qubit, line)
            if n_cavity is not None:
                line = re.sub(r'n_cavity\s*=\s*\d+',
                            r'n_cavity = %d' % n_cavity, line)
            if dissipation is not None:
                if dissipation:
                    val = 'T'
                    line = re.sub(r'prop\s*=\s*cheby', r'prop = newton', line)
                else:
                    val = 'F'
                    line = re.sub(r'prop\s*=\s*newton', r'prop = cheby', line)
                line = re.sub(r'dissipation\s*=\s*[TF]',
                              r'dissipation = %s' % val, line)
            out_fh.write(line)
    config_content = read_file(config)
    if rho:
        assert "rho_prop_mode" in config_content
        assert "gamma_phi_1" in config_content
        assert "gamma_phi_2" in config_content


def num_pulse_to_analytic(runfolder, formula, rwa=True, randomize=False,
        num_pulse='pulse.guess', analytical_pulse='pulse.json'):
    """Read a numerical guess pulse from `num_pulse` and map it as closely as
    possible to an analytical pulse, to be stored in `analytical_pulse`. The
    analytic pulse will have the given formula.

    If `randomize` is True, instead of trying to achieve a close fit between
    `num_pulse` and `analytical_pulse`, make a "trivial" guess for the
    `analytical_pulse` to make it roughtly equivalent to the `num_pulse` (e.g.,
    match the amplitude), and then apply a 10% random variation on all the
    analytic pulse parameters. This may be useful to try to get out of a local
    optimization minimum.

    If `num_pulse` does not exist, `analytical_pulse` is removed. If both
    `num_pulse` and `analytical_pulse` exist, and `analytical_pulse` is newer
    than `num_pulse`, all files are left untouched; if it is older, it is
    replaced by a new fit.
    """
    logger = logging.getLogger(__name__)
    pulse_guess = os.path.join(runfolder, num_pulse)
    pulse_json = os.path.join(runfolder, analytical_pulse)
    config = os.path.join(runfolder, 'config')
    # handle existing files
    try:
        p_guess = Pulse(filename=pulse_guess)
    except (IOError, ValueError) as exc_info:
        # p_guess does not exist or is unreadable
        if os.path.isfile(pulse_json):
            os.unlink(pulse_json)
        logger.error(str(exc_info.value))
        return
    if os.path.isfile(pulse_json):
        if (os.path.getctime(pulse_json) > os.path.getctime(pulse_guess)):
            # already matched
            return
    # perform the fit
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
    guess_analytical.write(pulse_json)
    logger.debug("Mapped %s to analytic %s: %s"
                % (num_pulse, analytical_pulse, guess_analytical.header))


def switch_to_analytical_guess(runfolder, num_guess='pulse.guess',
        analytical_guess='pulse_opt.json', backup='pulse.guess.pre_simplex',
        nt_min=2000):
    """Replace `num_guess` with `analytical_guess` (converted to a numeric
    pulse), while backing up the original `num_guess` to `backup`.

    If `num_guess` does not exist (but `analytical_guess` does), simply convert
    `analytical_guess` to `num_guess` without creating a backup)
    """
    pulse_guess             = os.path.join(runfolder, num_guess)
    pulse_opt_json          = os.path.join(runfolder, analytical_guess)
    pulse_guess_pre_simplex = os.path.join(runfolder, backup)
    config                  = os.path.join(runfolder, 'config')
    if not os.path.isfile(pulse_guess):
        if os.path.isfile(pulse_guess_pre_simplex):
            QDYN.shutil.copy(pulse_guess_pre_simplex, pulse_guess)
    if not os.path.isfile(pulse_opt_json):
        if os.path.isfile(pulse_guess_pre_simplex):
            QDYN.shutil.copy(pulse_guess_pre_simplex, pulse_guess)
    if os.path.isfile(pulse_guess):
        QDYN.shutil.copy(pulse_guess, pulse_guess_pre_simplex)
    if os.path.isfile(pulse_opt_json):
        p_guess = AnalyticalPulse.read(pulse_opt_json)
        if p_guess.nt < nt_min:
            p_guess.nt = nt_min
        if p_guess.formula_name == '1freq_rwa':
            assert p_guess.parameters['w_L'] == p_guess.parameters['w_d']
        p_guess.pulse().write(pulse_guess)
    pulse_config_compat(p_guess, config, adapt_config=True)


def systematic_scan(runfolder, template_pulse, scan_params_json,
        outfile='pulse_systematic_scan.json', target='target_gate.dat',
        rwa=False, use_threads=False):
    """Read a dictionary from the json file scan_params_json that must map
    ``param => array of values``, where `param` is a parameter in the
    analytical formula of the pulse defined in the json file `template_pulse`.

    Propagate for each possible value combination and write the pulse that
    yields the best figure of merit to `outfile`
    """
    if not rwa:
        raise NotImplementedError("LAB frame not supported")
    pulse0 = AnalyticalPulse.read(os.path.join(runfolder, template_pulse))
    with open(os.path.join(runfolder, scan_params_json)) as in_fh:
        vary = json.load(in_fh)
    def worker(args):
        rf, pulse_json = args
        U = propagate(rf, pulse_json, target=target, rwa=True, force=True,
                      keep=None)
        return U
    if target in ['PE', 'SQ']:
        U_tgt = None
    else:
        U_tgt = read_target_gate(os.path.join(runfolder, target))
    def fig_of_merit(U):
        if target in ['PE', 'SQ']:
            C = U.closest_unitary().concurrence()
            max_loss = np.max(1.0 - U.logical_pops())
            return J_target(target, C, max_loss)
        else:
            return 1.0 - U.F_avg(U_tgt)
    table = zeta_systematic_variation.systematic_variation(runfolder, pulse0,
            vary, fig_of_merit, n_procs=1, _worker=worker)
    pulse1 = pulse0.copy()
    row = table.iloc[0]
    for key in row.keys():
        pulse1.parameters[key] = row[key]
    pulse1.write(os.path.join(runfolder, outfile))


def run_pre_krotov_simplex(runfolder, formula_or_json_file, vary='default',
        target='target_gate.dat', rwa=False, randomize=False):
    """Run a simplex pre-optimization, resulting in file 'pulse_opt.json' in
    the runfolder. If `formula_or_json_file` is a formula, the starting point
    of the optimization is an analytic approximation to a numeric pulse in
    'pulse.guess', which will be written to 'pulse.json'. If
    `formula_or_json_file` is the name of a json file inside the runfolder, the
    analytic pulse described in that json file is the starting point for the
    optimization.

    The result of the simplex optimization will be written to pulse_opt.json

    If 'pulse_opt.json' already exists and is newer than the guess pulse file,
    nothing is done.
    """
    logger = logging.getLogger(__name__)
    guess = formula_or_json_file
    pulse_json = os.path.join(runfolder, guess)
    pulse_opt_json = os.path.join(runfolder, 'pulse_opt.json')
    config = os.path.join(runfolder, 'config')
    if not os.path.isfile(pulse_json):
        # We assume that a formula name was given
        formula = formula_or_json_file
        num_pulse_to_analytic(runfolder, formula, rwa,
                randomize, num_pulse='pulse.guess',
                analytical_pulse='pulse.json')
        guess = 'pulse.json'
    pulse_json = os.path.join(runfolder, guess)
    if os.path.isfile(pulse_opt_json):
        if (os.path.getctime(pulse_opt_json) > os.path.getctime(pulse_json)):
            logger.debug("%s already up to date" % pulse_opt_json)
            return
    # if we're going to do a new pre-simplex optimization, we have to
    # delete any files that will be generated by Krotov, as they are
    # now invalid (starting from the wrong guess pulse)
    for file in ['oct.log', 'pulse.dat', 'oct_iters.dat', 'config.oct',
            'U.dat', 'prop.log', 'simplex.log']:
        file = os.path.join(runfolder, file)
        if os.path.isfile(file):
            os.unlink(file)
    # run simplex optimization (pulse.json -> pulse_opt.json)
    if target in ['PE', 'SQ']:
        target_gate = target
        extra_files_to_copy=[]
    else:
        target_gate_dat = os.path.join(runfolder, target)
        target_gate = read_target_gate(target_gate_dat)
        extra_files_to_copy=[target]
    assert re.search(r'prop_guess\s*=\s*F', read_file(config))
    assert re.search(r'oct_outfile\s*=\s*pulse.dat', read_file(config))
    logger.debug("Running simplex to optimize for target %s" % target)
    run_simplex(runfolder, target=target_gate, rwa=rwa,
                prop_pulse_dat='pulse.dat',
                extra_files_to_copy=extra_files_to_copy,
                guess_pulse=guess, opt_pulse='pulse_opt.json', vary=vary,
                fixed_parameters=['T', 'w_d', 'freq_1', 'freq_2'])


def run_oct(runfolder, target='target_gate.dat', rwa=False,
        continue_oct=False, g_a_int_min_initial=1.0e-5, g_a_int_max=1.0e-1,
        g_a_int_converged=1.0e-7, iter_stop=None, J_T_re=False, lbfgs=False,
        use_threads=False):
    """Run optimal control on the given runfolder. Adjust lambda_a if
    necessary. Target may either be 'PE', 'SQ', or the name of file defining a
    gate, inside the runfolder.

    Assumes that the runfolder contains the files config and pulse.guess, the
    file defined by `target`, and optionally, pulse.dat, and oct_iters.dat.

    Creates (overwrites) the files pulse.dat and oct_iters.dat.

    Also, a file config.oct is created that contains the last update to
    lambda_a. The original config file will remain unchanged.
    """
    logger = logging.getLogger(__name__)
    temp_runfolder = get_temp_runfolder(runfolder)
    QDYN.shutil.mkdir(temp_runfolder)
    config = os.path.join(runfolder, 'config')
    temp_config = os.path.join(temp_runfolder, 'config')
    temp_pulse_dat = os.path.join(temp_runfolder, 'pulse.dat')
    write_oct_config(config, temp_config, target, iter_stop=iter_stop,
                     J_T_re=J_T_re, lbfgs=lbfgs)
    required_files = ['pulse.guess']
    if target not in ['PE', 'SQ']:
        required_files.append(target)
    files_to_copy = list(required_files)
    if continue_oct:
        files_to_copy.extend(['pulse.dat', 'oct_iters.dat'])
    for file in files_to_copy:
        if os.path.isfile(os.path.join(runfolder, file)):
            QDYN.shutil.copy(os.path.join(runfolder, file), temp_runfolder)
            logger.debug("%s to temp_runfolder %s", file, temp_runfolder)
        else:
            if file in required_files:
                raise IOError("%s does not exist in %s" % (file, runfolder))
    logger.info("Starting optimization of %s (in %s)", runfolder,
                temp_runfolder)
    with open(os.path.join(runfolder, 'oct.log'), 'w', 0) as stdout:
        ensure_ham_files(temp_runfolder, stdout=stdout)
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
            env = os.environ.copy()
            env['OMP_NUM_THREADS'] = '1'
            if use_threads:
                env['OMP_NUM_THREADS'] = '4'
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
                    if ( (iter == 1)
                    and (g_a_int < g_a_int_min_initial)
                    and (not pulse_explosion) ):
                        logger.debug("pulse update too small: %g < %g"
                                     % (g_a_int, g_a_int_min_initial))
                        logger.debug("Kill %d" % oct_proc.pid)
                        oct_proc.kill()
                        scale_lambda_a(temp_config, 0.5)
                        reset_pulse(temp_pulse_dat, iter)
                        break # next bad_lambda loop
                    # if the pulse update explodes, we increase lambda_a (and
                    # prevent it from decreasing again)
                    if ( ('amplitude exceeds maximum value' in line)
                    or   ('Loss of monotonic convergence' in line)
                    or   (g_a_int > g_a_int_max) ):
                        pulse_explosion = True
                        if "Loss of monotonic convergence" in line:
                            logger.debug("loss of monotonic conversion")
                        else:
                            if (g_a_int > g_a_int_max):
                                logger.debug("g_a_int = %g > %g",
                                              g_a_int, g_a_int_max)
                            logger.debug("pulse explosion")
                        logger.debug("Kill %d" % oct_proc.pid)
                        oct_proc.kill()
                        scale_lambda_a(temp_config, 1.25)
                        reset_pulse(temp_pulse_dat, iter)
                        break # next bad_lambda loop
                    # if there are no significant pulse changes anymore, we
                    # stop the optimization prematurely
                    if iter > 10 and g_a_int < g_a_int_converged:
                        logger.debug(("pulse update insignificant "
                            "(converged): g_a_int = %g < %g")
                            % (g_a_int, g_a_int_converged))
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


def propagate(runfolder, pulse_file, rwa, rho=False, rho_pop_plot=False,
        n_qubit=None, n_cavity=None, dissipation=True, keep=False, force=False,
        target='target_gate.dat', use_threads=False):
    """Map runfolder -> 2QGate, by propagating the given pulse file, or reading
    from an existing U.dat (if force is False)

    The file indicated by `pulse_file` may contain either a numerical pulse,
    or, if it has a 'json' extension, and analytic pulse.

    If `rho` is True, propagate in Liouville space. Note that in this case, the
    returned 2QGate is a unitary approximation of the dynamics. The exact value
    of F_avg should be extracted from the 'prop.log' file resulting from the
    propagation. The `rho_pop_plot` option may be used (in combination with
    rho=True and keep=True) to produce the files necessary for a population
    plot. Note that with this option, the propage average gate fidelity can not
    be extracted from the prop.log file.

    By giving the n_qubit and/or n_cavity options, the number of qubit and
    cavity levels can be changed from the value given in the config file.

    If `dissipation` is given as False, the corresponding `dissipation` value
    is set to False in the config, file, resulting in unitary dynamics.

    If `keep` is True, keep all files resulting from the propagation in the
    runfolder. Otherwise, only prop.log and U.dat will be kept. Note that
    the original config file is never overwritten, but the config file from the
    temporary propagation folder (which was possibly modified due to the
    `pulse_file`, `rho_pop_plot`, `n_qubit`, or `n_cavity` options) is copied
    in as ``config.prop``, if `keep` is True.

    If `keep` is None, no files in the runfolder will be modified.

    If use_threads is True, use 4 OpenMP threads
    """
    logger = logging.getLogger(__name__)
    if n_qubit is not None:
        n_qubit = int(n_qubit)
        assert n_qubit > 2
    if n_cavity is not None:
        n_cavity = int(n_cavity)
        assert n_cavity > 1
    gatefile = os.path.join(runfolder, 'U.dat')
    config = os.path.join(runfolder, 'config')
    target_gate_dat = os.path.join(runfolder, target)
    if os.path.isfile(gatefile) and force:
        try:
            os.unlink(gatefile)
        except OSError:
            pass
    U = None
    if (not os.path.isfile(gatefile)) or force:
        temp_config = None
        try:
            temp_runfolder = get_temp_runfolder(runfolder)
            temp_config = os.path.join(temp_runfolder, 'config')
            temp_U_dat = os.path.join(temp_runfolder, 'U.dat')
            logger.debug("Prepararing temp_runfolder %s", temp_runfolder)
            QDYN.shutil.mkdir(temp_runfolder)
            if os.path.isfile(target_gate_dat):
                QDYN.shutil.copy(target_gate_dat, temp_runfolder)
            # copy over the pulse
            analytical_pulse = None
            if pulse_file.endswith(".json"):
                analytical_pulse = AnalyticalPulse.read(
                                   os.path.join(runfolder, pulse_file))
                pulse = analytical_pulse.pulse()
            else:
                pulse = Pulse(os.path.join(runfolder, pulse_file))
            pulse.write(os.path.join(temp_runfolder, 'pulse_prop.dat'))
            # copy over the config file, with modifications
            write_prop_config(config, temp_config, 'pulse_prop.dat',
                              rho=rho, rho_pop_plot=rho_pop_plot,
                              n_qubit=n_qubit, n_cavity=n_cavity,
                              dissipation=dissipation)
            if analytical_pulse is not None:
                pulse_config_compat(analytical_pulse, temp_config,
                                    adapt_config=True)
            logger.info("Propagating %s", runfolder)
            env = os.environ.copy()
            env['OMP_NUM_THREADS'] = '1'
            if use_threads:
                env['OMP_NUM_THREADS'] = '4'
            start = time.time()
            prop_log = os.path.join(runfolder, 'prop.log')
            if keep is None:
                prop_log = os.path.join(temp_runfolder, 'prop.log')
            with open(prop_log, 'w', 0) as stdout:
                ensure_ham_files(temp_runfolder, rwa, stdout, rho, dissipation)
                cmds = []
                if rho:
                    cmds.append(['tm_en_rho_prop', '.'])
                else:
                    cmds.append(['tm_en_prop', '.'])
                for cmd in cmds:
                    stdout.write("**** " + " ".join(cmd) +"\n")
                    sp.call(cmd , cwd=temp_runfolder, env=env,
                            stderr=sp.STDOUT, stdout=stdout)
            U = QDYN.gate2q.Gate2Q(file=temp_U_dat)
            if keep is not None:
                QDYN.shutil.copy(temp_U_dat, runfolder)
            end = time.time()
            logger.info("Finished propagating %s (%d seconds)",
                         runfolder, end-start)
        except Exception as e:
            logger.error(e)
        finally:
            if keep is None:
                keep = False
            if keep:
                if temp_config is not None and os.path.isfile(temp_config):
                    os.rename(temp_config, temp_config + ".prop")
                sp.call(['rsync', '-a', '%s/'%temp_runfolder, runfolder])
            QDYN.shutil.rmtree(temp_runfolder)
    else:
        logger.info("Propagating of %s skipped (gatefile already exists)",
                     runfolder)
    # If we performed a propagation, U should already have been set above based
    # on "U.dat" in the temporary folder. If we're loading from an existing
    # U.dat intead, we do that here.
    if U is None:
        try:
            U = QDYN.gate2q.Gate2Q(file=gatefile)
        except (IOError, OSError) as e:
            logger.error(e)
    if np.isnan(U).any():
        logger.error("gate %s contains NaN", gatefile)
    return U


def get_iter_stop(config):
    """Extract the value of iter_stop from the given config file"""
    with open(config) as in_fh:
        for line in in_fh:
            m = re.search(r'iter_stop\s*=\s*(\d+)', line)
            if m:
                return int(m.group(1))
    return None


@click.command(help=__doc__)
@click.help_option('--help', '-h')
@click.option('--target',  metavar='TARGET', default='target_gate.dat',
    show_default=True,
    help="Optimization target. Can be 'PE', 'SQ', or the name of a gate "
    "file inside the runfolder.")
@click.option('--J_T_re', 'J_T_re', is_flag=True, default=False,
    help='If TARGET is a gate file, use a phase sensitive functional '
    'instead of the default square-modulus functional')
@click.option('--lbfgs', is_flag=True, default=False,
    help='If TARGET is a gate file, use the lbfgs optimization method. '
    'Implies --J_T_re')
@click.option('--rwa', is_flag=True, default=False,
    help="Perform all calculations in the RWA.")
@click.option('--continue', 'cont', is_flag=True, default=False,
    help="Continue from an existing pulse.dat")
@click.option( '--debug', is_flag=True, default=False,
    help="Enable debugging output")
@click.option('--threads', 'use_threads', is_flag=True, default=False,
    help="Use 4 OpenMP threads")
@click.option('--prop-only', is_flag=True, default=False,
    help="Only propagate, instead of doing OCT")
@click.option(
    '--prop-rho', is_flag=True, default=False,
    help="Do the propagation in Liouville space.")
@click.option('--prop-n_qubit',  type=int,
    help="In the (post-oct) propagation, use the given "
    "number of qubit levels, instead of the number specified in the "
    "config file. Does not affect OCT.")
@click.option('--prop-n_cavity', type=int,
    help="In the (post-OCT) propagation, use the given "
    "number of cavity levels, instead of the number specified in the "
    "config file. Does not affect OCT.")
@click.option('--rho-pop-plot', is_flag=True, default=False,
    help="In combination with --prop-rho and --keep, "
    "produce a population plot")
@click.option('--keep', is_flag=True, default=False,
    help="Keep all files from the propagation")
@click.option('--pre-simplex', 'formula_or_json_file',
    help="Run simplex pre-optimization before Krotov. Parameter may "
    "either be the name of a pulse formula, or the name of a json file. "
    "If it is a formula name, an analytic approximation to the existing "
    "file 'pulse.guess' will be the guess pulse for the simplex "
    "optimization. If it is a json file, then the analytic pulse "
    "described in that file will be the guess pulse.")
@click.option('--vary', multiple=True,
    help='If given in conjunction with '
    '--pre-simplex, the parameter that will be varied in the simplex '
    'search. Can be given multiple times to vary more than one parameter. '
    'If not given, the parameters to be varied are chosen automatically')
@click.option('--scan', metavar='SCAN_PARAMS_JSON',
    type=click.Path(exists=True, dir_okay=False),
    help="If given in conjunction with --pre-simplex, perform a systematic "
    "scan of parameters before doing the simplex-pre-optimization. The file "
    "SCAN_PARAMS_JSON must be a json dump of a dictionary that maps parameter "
    "names to values to try. All possible combinations will be tried, and the "
    "one with the best figure of merit will be the starting point for the "
    "simplex optimization")
@click.option('--randomize', is_flag=True, default=False,
    help="In combination with --pre-simplex, start from "
    "a randomized analytic guess pulse, instead of one matching the "
    "original numerical pulse.guess as closely as possible.")
@click.option('--g_a_int_min_initial',  default=1.0e-5, type=float,
    help="The smallest acceptable value "
    "for g_a_int in the first OCT iteration. For any smaller value,"
    "lambda_a is deemed too big, and will be adjusted.")
@click.option('--g_a_int_max',  default=1.0e-1, type=float,
    help="The largest acceptable value for "
    "g_a_int. Any larger value is taken as a 'pulse explosion', "
    "requiring lambda_a to be increased.")
@click.option('--g_a_int_converged',  default=1.0e-7, type=float,
    help="The smallest value for g_a_int "
    "before the optimization is assumed to be converged.")
@click.option('--iter_stop',  type=int,
    help="The iteration number after which to stop OCT")
@click.option('--nt-min',  default=2000, type=int,
    help="The minimum nt to be used when converting an "
    "analytical pulse to a numerical one.")
@click.argument('runfolder', type=click.Path(exists=True, dir_okay=True,
    file_okay=False))
def main(target, J_T_re, lbfgs, rwa, cont, debug, use_threads, prop_only,
        prop_rho, prop_n_qubit, prop_n_cavity, rho_pop_plot, keep,
        formula_or_json_file, vary, scan, randomize, g_a_int_min_initial,
        g_a_int_max, g_a_int_converged, iter_stop, nt_min, runfolder):
    assert 'SCRATCH_ROOT' in os.environ, \
    "SCRATCH_ROOT environment variable must be defined"
    if iter_stop is None:
        iter_stop = get_iter_stop(os.path.join(runfolder, 'config'))
    pulse_file = (os.path.join(runfolder, 'pulse.dat'))
    logger = logging.getLogger()
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    if prop_only:
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
        if formula_or_json_file is not None:
            if os.path.isfile(pulse_file) and cont:
                logger.debug("Skip simplex, continuing existing pulse")
            else:
                if len(vary) == 0:
                    vary = 'default'
                if scan is None:
                    # {formula_or_json_file} -> pulse_opt.json
                    run_pre_krotov_simplex(runfolder, formula_or_json_file,
                            vary=vary, target=target, rwa=rwa,
                            randomize=randomize)
                else:
                    # {formula_or_json_file} -> pulse_systematic_scan.json
                    if os.path.isfile(
                            os.path.join(runfolder, formula_or_json_file)):
                        systematic_scan(runfolder, formula_or_json_file, scan,
                                outfile='pulse_systematic_scan.json',
                                target=target, rwa=rwa,
                                use_threads=use_threads)
                    else:
                        raise NotImplementedError("Scan is implemented only "
                                "for starting from an analytic pulse file, "
                                "not from a formula")
                    # pulse_systematic_scan.json -> pulse_opt.json
                    run_pre_krotov_simplex(runfolder,
                            'pulse_systematic_scan.json',
                            vary=vary, target=target, rwa=rwa,
                            randomize=randomize)
                switch_to_analytical_guess(runfolder, num_guess='pulse.guess',
                    analytical_guess='pulse_opt.json',
                    backup='pulse.guess.pre_simplex', nt_min=nt_min)
        if os.path.isfile(os.path.join(runfolder, 'U.dat')):
            # if we're doing a new oct, we should delete U.dat
            os.unlink(os.path.join(runfolder, 'U.dat'))
        if os.path.isfile(os.path.join(runfolder, 'U_closest_PE.dat')):
            os.unlink(os.path.join(runfolder, 'U_closest_PE.dat'))
        run_oct(runfolder, target=target, rwa=rwa, continue_oct=cont,
                g_a_int_min_initial=g_a_int_min_initial,
                g_a_int_max=g_a_int_max,
                g_a_int_converged=g_a_int_converged,
                iter_stop=iter_stop, J_T_re=J_T_re,
                lbfgs=lbfgs, use_threads=use_threads)
    if not os.path.isfile(os.path.join(runfolder, 'U.dat')):
        propagate(runfolder, 'pulse.dat', rwa=rwa, rho=prop_rho,
                  rho_pop_plot=rho_pop_plot, n_qubit=prop_n_qubit,
                  n_cavity=prop_n_cavity, keep=keep, target=target,
                  use_threads=use_threads)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
