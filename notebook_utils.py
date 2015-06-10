"""
Module containing auxilliary routines for the notebooks in this folder
"""
import QDYN
from glob import glob
import re
import os
import numpy as np
from matplotlib.mlab import griddata
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
from mgplottools.mpl import get_color, set_axis, new_figure

def get_field_free_data(runs):
    """Return 3 numpy arrays: w_2, w_c, C, loss"""
    rx_folder = re.compile(r'w2_(\d+)MHz_wc_(\d+)MHz/')
    w2s    = []
    wcs    = []
    Cs     = []
    losses = []
    with QDYN.shutil.chdir(runs):
        folders = glob('*/')
        for folder in folders:
            m = rx_folder.match(folder)
            if m:
                w2 = float(m.group(1)) / 1000.0
                wc = float(m.group(2)) / 1000.0
                U_file = os.path.join(folder, 'stage1', 'field_free', 'U.dat')
                if os.path.isfile(U_file):
                    U = QDYN.gate2q.Gate2Q(U_file)
                else:
                    continue
                C = U.closest_unitary().concurrence()
                loss = U.pop_loss()
                w2s.append(w2)
                wcs.append(wc)
                Cs.append(C)
                losses.append(loss)
    return np.array(w2s), np.array(wcs), np.array(Cs), np.array(losses)


def plot_field_free_data(runs):
    """Plot field-free concurrence"""
    plot_width      =  16.0
    left_margin     =  1.0
    top_margin      =  1.0
    bottom_margin   =  1.0
    h               =  10.0
    w               =  10.0
    cbar_width      =  0.3
    cbar_gap        =  0.5
    h_offset        =  17.0

    fig_width = h_offset + 2*plot_width
    fig_height = bottom_margin + h + top_margin

    w_2, w_c, C, loss = get_field_free_data(runs)

    fig = new_figure(fig_width, fig_height, quiet=True)

    pos_contour = [left_margin / fig_width, bottom_margin / fig_width,
                   w/fig_width, h/fig_height]
    ax_contour = fig.add_axes(pos_contour)
    pos_cbar = [(left_margin + w + cbar_gap) / fig_width,
                bottom_margin/fig_width, cbar_width/fig_width, h/fig_height]
    ax_cbar = fig.add_axes(pos_cbar)
    render_values(w_2, w_c, C, fig, ax_contour, ax_cbar, vmin=0.0, vmax=1.0)
    ax_contour.set_title("concurrence")

    pos_contour = [(left_margin+h_offset)/fig_width, bottom_margin/fig_width,
                   w/fig_width, h/fig_height]
    ax_contour = fig.add_axes(pos_contour)
    pos_cbar = [(left_margin+w+cbar_gap+h_offset)/fig_width,
                bottom_margin/fig_width, cbar_width/fig_width, h/fig_height]
    ax_cbar = fig.add_axes(pos_cbar)
    render_values(w_2, w_c, loss, fig, ax_contour, ax_cbar, logscale=True,
                  vmin=1e-3, vmax=1.0)
    ax_contour.set_title("population loss")

    plt.show(fig)


def find_files(directory, pattern):
    """
    Iterate (recursively) over all the files matching the shell pattern
    ('*' will yield all files) in the given directory
    """
    import fnmatch
    if not os.path.isdir(directory):
        raise IOError("directory %s does not exist" % directory)
    for root, __, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename

def find_folders(directory, pattern):
    """
    Iterate (recursively) over all the folders matching the shell pattern
    ('*' will yield all folders) in the given directory
    """
    import fnmatch
    if not os.path.isdir(directory):
        raise IOError("directory %s does not exist" % directory)
    for root, dirs, __ in os.walk(directory):
        for basename in dirs:
            if fnmatch.fnmatch(basename, pattern):
                foldername = os.path.join(root, basename)
                yield foldername


def render_values(w_2, w_c, val, fig, ax_contour, ax_cbar, density=100,
    logscale=False, vmin=None, vmax=None, n_contours=10):
    x = np.linspace(w_2.min(), w_2.max(), density)
    y = np.linspace(w_c.min(), w_c.max(), density)
    z = griddata(w_2, w_c, val, x, y, interp='linear')
    if vmin is None:
        vmin=abs(z).min()
    if vmax is None:
        vmax=abs(z).max()
    if logscale:
        contours = ax_contour.pcolormesh(x, y, z, cmap=plt.cm.gnuplot2,
                                         norm=LogNorm(), vmax=vmax, vmin=vmin)
    else:
        if n_contours > 0:
            ax_contour.contour(x, y, z, n_contours, linewidths=0.5, colors='k')
        contours = ax_contour.pcolormesh(x, y, z, cmap=plt.cm.gnuplot2,
                                    vmax=abs(z).max(), vmin=abs(z).min())
    ax_contour.scatter(w_2, w_c, marker='o', c='cyan', s=5, zorder=10)
    ax_contour.set_xlabel(r"$\omega_2$ (GHz)")
    ax_contour.set_ylabel(r"$\omega_c$ (GHz)")
    cbar = fig.colorbar(contours, cax=ax_cbar)


def cutoff_worker(x):
    """
    Map w_L [GHz], E_0 [MHz], nt, n_q, n_c -> 2QGate

    Used in Prereq_Cutoff.ipynb
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

