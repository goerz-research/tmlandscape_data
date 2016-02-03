#!/usr/bin/env python
"""Create a set of runfolders inside the `runs` folder, based on table with
columns

    w2, wc, wd, T, target,

in the `parameters_file`, where w2, wc, wd are the (2nd) qubit frequency,
cavity frequency, and rotating frame frequency, in GHz, T is the gate duration
in ns and target is one of 'PE', 'SQ', 'H_left', 'H_right', 'Ph_left',
'Ph_right'. RWA is implied.
"""
import sys
import os
from select_for_stage4 import GATE_RE, GATE_IM
from textwrap import dedent
from mgplottools.io import writetotxt
import numpy as np
import QDYN
from QDYN.pulse import Pulse, pulse_tgrid
import logging

# we can handle optimization towards the following targets:
TARGETS = ['PE', 'SQ', 'H_left', 'H_right', 'Ph_left', 'Ph_right']
SQ_TARGETS = ['H_left', 'H_right', 'Ph_left', 'Ph_right']


def write_config(config_file, T, nt, wc, w2, wd, gate="target_gate.dat",
        J_T='SM', prop_guess='F'):
    """Write out the config file (in the RWA)

    Arguments:
        config_file (str): path to the config file to write
        T (float): gate duration in ns
        wc (float): cavity frequency in GHz
        w2 (float): 2nd qubit frequency in GHz
        wd (float): frequency of rotating frame, in GHz
        gate (str): Gate to optimize for
        J_T (str): functional to use
    """
    config = dedent(r'''
    tgrid: n = 1
    1 : t_start = 0.0, t_stop = {T:.4f}_ns, nt = {nt:d}

    pulse: n = 1
    1: type = file, filename = pulse.guess, id = 1,  time_unit = ns, &
    ampl_unit = MHz, is_complex = T, oct_increase_factor = 5.0, &
    oct_outfile = pulse.dat, oct_lambda_a = 1.0e-1, oct_lambda_intens = 0.0, &
    oct_shape = flattop, shape_t_start = 0.0, t_rise = {t_rise_fall:.4f}_ns, &
    shape_t_stop = {T:.4f}_ns, t_fall = {t_rise_fall}_ns, check_tgrid = F

    oct: iter_stop = 10000, max_megs = 9000, type = krotovpk, &
        A = 0, B = 0, C = 0.0, iter_dat = oct_iters.dat, &
        keep_pulses = prev, max_hours = 23,  continue = T, dynamic_sigma = T, &
        sigma_form = local, J_T_conv = 1.0e-3

    misc: prop = newton, mass = 1.0

    user_ints: n_qubit = 5, n_cavity = 6

    user_strings: gate = {gate}, J_T = {J_T}

    user_logicals: prop_guess = {prop_guess}, dissipation = T

    user_reals: &
    LI_unitarity_weight = 0.01, &
    w_c     = {w_c}_MHz, &
    w_1     = 6000.0_MHz, &
    w_2     = {w_2}_MHz, &
    w_d = {w_d}_MHz, &
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
    with open(config_file, 'w')  as out_fh:
        out_fh.write(config.format(
            T=T, nt=nt, w_c=(float(wc)*1000.0),
            w_2=(float(w2)*1000.0), w_d=(float(wd)*1000.0),
            t_rise_fall=min(2, 0.1*float(wd)*1000.0), gate=gate, J_T=J_T,
            prop_guess=prop_guess))

def generate_folder(w2, wc, wd, T, runs, strategy, target, w_max=1.0,
        dry_run=False):
    """Generate a runfolder for optimization at a given paramter point towards
    a given target, in the RWA

    Paramters:
        w2 (float): 2nd qubit frequency, in GHz
        wc (float): cavity frequency, in GHz
        wd (float): drive frequency, in GHz
        T (float): gate duration, in ns
        runs (str): the root under which the runfolder should be generated
        strategy (str): name of a subfolder, for differentiating between
            OCT strategies (see below)
        target (str): one of those listed in TARGETS
        w_max (float): minimum frequency in the rotating frame to be resolved.
            This determines the number of time grid points
        dry_run (boolean): If True, don't generate any files, only log info

    The runfolder is created as

        {runs}/w2_{w2}MHz_wc_{wc}MHz/{strategy}/{target}

    w1 = 6.0 is implicit. A zero-amplitude pulse is written to 'pulse.guess' in
    the runfolder. Furthormoe, 'target_gate.dat' is written to the runfolder.
    """
    logger = logging.getLogger(__name__)
    w2_wc_folder = 'w2_%dMHz_wc_%dMHz' % (w2*1000, wc*1000)
    assert target in TARGETS
    runfolder = os.path.join(runs, w2_wc_folder, strategy, target)
    nt_rwa = int(max(2000, 100 * w_max * T))
    if dry_run:
        logger.info("Creating %s", runfolder)
    else:
        QDYN.shutil.mkdir(runfolder)
    config_file = os.path.join(runfolder, 'config')
    logger.info("Writing %s", config_file)
    if dry_run:
        logger.info("Writing %s", config_file)
    else:
        write_config(config_file, T, nt_rwa, wc, w2, wd)
    pulse_guess = os.path.join(runfolder, 'pulse.guess')
    if dry_run:
        logger.info("Writing %s", pulse_guess)
    else:
        p = Pulse(tgrid=pulse_tgrid(T, nt_rwa), time_unit='ns',
                ampl_unit='MHz')
        p.write(pulse_guess)
    if target in SQ_TARGETS:
        target_gate_dat = os.path.join(runfolder, 'target_gate.dat')
        if dry_run:
            logger.info("Writing %s", target_gate_dat)
        else:
            writetotxt(target_gate_dat,
                    GATE_RE[target], GATE_IM[target])
    else:
        raise NotImplementedError


def process_parameters_file(parameters_file, runs, strategy, dry_run=False):
    """For every line in the given `parameters_file`, call the
    `generate_folder` routine
    """
    w2_s, wc_s, wd_s, T_s, targets = np.genfromtxt(parameters_file,
                                                   unpack=True)
    for (w2, wc, wd, T, target) in zip(w2_s, wc_s, wd_s, T_s, targets):
        for target in SQ_TARGETS:
            generate_folder(w2, wc, wd, T, runs, strategy, target,
                    dry_run=dry_run)


def main(argv=None):
    """Main routine"""
    from optparse import OptionParser
    logger = logging.getLogger(__name__)
    if argv is None:
        argv = sys.argv
    arg_parser = OptionParser(
    usage = "%prog [options] <runs> <parameters_file>",
    description = __doc__)
    arg_parser.add_option(
        '--debug', action='store_true', dest='debug',
        default=False, help="Enable debugging output")
    arg_parser.add_option(
        '-n', action='store_true', dest='dry_run',
        help="Perform a dry run")
    arg_parser.add_option(
        '--strategy', action='store', dest='strategy',
        default='oct', help="Name of 'strategy' folder. Alternative strategy "
        "folder names may be used to explore different OCT strategies")
    options, args = arg_parser.parse_args(argv)
    try:
        runs = args[1]
        parameters_file = args[2]
    except IndexError:
        arg_parser.error("Missing arguments")
    if options.debug:
        logger.setLevel(logging.DEBUG)
    process_parameters_file(parameters_file, runs, options.strategy,
            dry_run=options.dry_run)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
