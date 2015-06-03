"""
Module containing auxilliary routines for the notebooks in this folder
"""

def cutoff_worker(x):
    """
    Map w_L [GHz], E_0 [MHz], nt, n_q, n_c -> 2QGate
    """

    import QDYN
    from QDYN.pulse import Pulse, pulse_tgrid, blackman, carrier
    from QDYNTransmonLib.prop import propagate
    import os
    import shutil
    from textwrap import dedent

    w_L, E_0, nt, n_q, n_c = x

    CONFIG = dedent(r'''
    tgrid: n = 1
    1 : t_start = 0.0, t_stop = 200_ns, nt = {nt}

    pulse: n = 1
    1: type = file, filename = pulse.guess, id = 1, &
    oct_increase_factor = 5.0, oct_outfile = pulse.dat, oct_lambda_a = 1.0e6, time_unit = ns, ampl_unit = MHz, &
    oct_shape = flattop, t_rise = 10_ns, t_fall = 10_ns, is_complex = F

    oct: iter_stop = 10000, max_megs = 2000, type = krotovpk, A = 0.0, B = 0, C = 0.0, iter_dat = oct_iters.dat, &
        keep_pulses = all, max_hours = 11, delta_J_conv = 1.0e-8, J_T_conv = 1.0d-4, strict_convergence = T, &
        continue = T, params_file = oct_params.dat

    misc: prop = newton, mass = 1.0

    user_ints: n_qubit = {n_q}, n_cavity = {n_c}

    user_strings: gate = CPHASE, J_T = SM

    user_logicals: prop_guess = T, dissipation = T

    user_reals: &
    w_c     = 10100.0_MHz, &
    w_1     = 6000.0_MHz, &
    w_2     = 6750.0_MHz, &
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
    ''')

    def write_run(config, params, pulse, runfolder):
        """Write config file and pulse to runfolder"""
        with open(os.path.join(runfolder, 'config'), 'w') as config_fh:
            config_fh.write(config.format(**params))
        pulse.write(filename=os.path.join(runfolder, 'pulse.guess'))

    commands = dedent(r'''
    export OMP_NUM_THREADS=4
    tm_en_gh --dissipation .
    rewrite_dissipation.py
    tm_en_prop . | tee prop.log
    ''')

    params = {'nt': nt, 'n_q': n_q, 'n_c': n_c}

    name = 'w%3.1f_E%03d_nt%d_nq%d_nc%d' % x
    gatefile = os.path.join('.', 'test_cutoff', 'U_%s.dat'%name)
    logfile = os.path.join('.', 'test_cutoff', 'prop_%s.log'%name)
    if os.path.isfile(gatefile):
        return QDYN.gate2q.Gate2Q(file=gatefile)
    runfolder = os.path.join('.', 'test_cutoff', name)
    QDYN.shutil.mkdir(runfolder)

    tgrid = pulse_tgrid(200.0, nt=nt)
    pulse = Pulse(tgrid=tgrid, time_unit='ns', ampl_unit='MHz')
    pulse.preamble = ['# Guess pulse: Blackman with E0 = %d MHz' % E_0]
    pulse.amplitude = E_0 * blackman(tgrid, 0, 200.0) \
                          * carrier(tgrid, 'ns', w_L, 'GHz')

    U = propagate(write_run, CONFIG, params, pulse, commands, runfolder)
    shutil.copy(os.path.join(runfolder, 'U.dat'), gatefile)
    shutil.copy(os.path.join(runfolder, 'prop.log'), logfile)
    shutil.rmtree(runfolder, ignore_errors=True)

    return U

