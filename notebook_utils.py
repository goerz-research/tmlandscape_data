"""
Module containing auxilliary routines for the notebooks in this folder
"""
import QDYN
import re
import os
import numpy as np
from matplotlib.mlab import griddata
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
from collections import OrderedDict
from mgplottools.mpl import set_axis, new_figure
from matplotlib import rcParams
#rcParams['xtick.direction'] = 'in'
#rcParams['ytick.direction'] = 'in'

###############################################################################
#  General Tools                                                              #
###############################################################################


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


def find_leaf_folders(directory):
    """
    Iterate (recursively) over all subfolders of the given directory that do
    not themselves contain subdirectories.
    """
    if not os.path.isdir(directory):
        raise IOError("directory %s does not exist" % directory)
    for root, dirs, __ in os.walk(directory):
        if not dirs:
            yield root


class PlotGrid(object):
    """Collection of colormap plots, layed out on a grid of cells

    Attributes
    ----------

    cell_width: float
        width of each panel area [cm]
    left_margin: float
        distance from left cell edge to axes [cm]
    top_margin: float
        distance from top cell edge to axes [cm]
    h: float
        height of axes [cm]
    w: float
        width of axes [cm]
    cbar_width: float
        width of colorbar [cm]
    cbar_gap: float
        gap between right edge of axes and colorbar [cm]
    density: int
        Number of interpolated points on each axis (resolution of color plot)
    contour_labels: boolean
        Whether or not to label contour lines in the plot
    """
    def __init__(self):
        self.cell_width      =  16.0
        self.left_margin     =  1.0
        self.top_margin      =  1.0
        self.bottom_margin   =  1.0
        self.h               =  10.0
        self.w               =  10.0
        self.cbar_width      =  0.3
        self.cbar_gap        =  0.5
        self.density         =  100
        self.n_cols          =  2
        self.contour_labels  = False
        self._cells          = [] # array of cell_dicts

    def add_cell(self, w2, wc, val, logscale=False, vmin=None, vmax=None,
        contour_levels=11, title=None):
        """Add a cell to the plot grid

        All other parameters will be passed to the render_values routine
        """
        cell_dict = {}
        cell_dict['w2'] = w2
        cell_dict['wc'] = wc
        cell_dict['val'] = val
        cell_dict['logscale'] = logscale
        if vmin is None:
            cell_dict['vmin'] = np.min(val)
        else:
            cell_dict['vmin'] = vmin
        if vmax is None:
            cell_dict['vmax'] = np.max(val)
        else:
            cell_dict['vmax'] = vmax
        cell_dict['contour_levels'] = contour_levels
        cell_dict['title'] = title
        self._cells.append(cell_dict)

    def plot(self, quiet=True, show=True):

        n_cells = len(self._cells)
        assert n_cells > 0, "No cells to plot"
        fig_width = self.n_cols * self.cell_width
        n_cols = self.n_cols
        n_rows = n_cells // n_cols
        if n_rows * n_cols < n_cells:
            n_rows += 1
        cell_width = self.cell_width
        cell_height = self.bottom_margin + self.h + self.top_margin
        fig_height = n_rows * cell_height
        fig = new_figure(fig_width, fig_height, quiet=quiet)

        col = 0
        row = 0

        for i, cell_dict in enumerate(self._cells):

            pos_contour = [(col*cell_width + self.left_margin)/fig_width,
                           ((n_rows-row-1)*cell_height
                            + self.bottom_margin)/fig_height,
                            self.w/fig_width, self.h/fig_height]
            ax_contour = fig.add_axes(pos_contour)

            pos_cbar = [(col*cell_width + self.left_margin + self.w
                         + self.cbar_gap)/fig_width,
                        ((n_rows-row-1)*cell_height
                            + self.bottom_margin)/fig_height,
                            self.cbar_width/fig_width, self.h/fig_height]
            ax_cbar = fig.add_axes(pos_cbar)

            render_values(cell_dict['w2'], cell_dict['wc'], cell_dict['val'],
                          ax_contour, ax_cbar, density=self.density,
                          logscale=cell_dict['logscale'],
                          vmin=cell_dict['vmin'], vmax=cell_dict['vmax'],
                          contour_levels=cell_dict['contour_levels'],
                          contour_labels=self.contour_labels)

            # show the resonance line (cavity on resonace with qubit 2)
            ax_contour.plot(np.linspace(5.0, 11.1, 10),
                            np.linspace(5.0, 11.1, 10), color='white')
            ax_contour.plot(np.linspace(6.0, 7.5, 10),
                            6.0*np.ones(10), color='white')
            ax_contour.axvline(6.29, color='white', ls='--')
            ax_contour.axvline(6.31, color='white', ls='--')
            ax_contour.axvline(6.58, color='white', ls='--')
            ax_contour.axvline(6.62, color='white', ls='--')
            # ticks and axis labels
            set_axis(ax_contour, 'x', 6.0, 7.5, 0.5, range=(6.1, 7.5), minor=5)
            set_axis(ax_contour, 'y', 5.0, 11.1, 0.5, minor=5)
            ax_contour.tick_params(which='both', direction='out')

            if cell_dict['title'] is not None:
                ax_contour.set_title(cell_dict['title'])

            col += 1
            if col == self.n_cols:
                col = 0
                row += 1

        if show:
            plt.show(fig)
        else:
            return fig


def render_values(w_2, w_c, val, ax_contour, ax_cbar, density=100,
    logscale=False, vmin=None, vmax=None, contour_levels=11,
    contour_labels=False):
    """Render the given data onto the given axes

    Parameters
    ----------

    w_2: array
        Array of w_2 values, in GHz (x-axis)
    w_c: array
        Array of w_c values, in GHz (y-axis)
    val: array
        Array of values (z-axis)
    ax_contour: matplotlib.axes.Axes instance
        Axes onto which to render the contour plot for (w_2, w_c, val)
    ax_cbar: matplotlib.axes.Axes instance
        Axes onto which to render the color bar describing the contour plot
    vmin: float
        Bottom value of the z-axis (colorbar range). Defaults to the minimum
        value in the val array
    vmax: float
        Top value of the z-axis (colorbar range). Defaults to the maximum
        value in the val array
    contour_levels: int, array of floats
        Contour lines to draw. If given as an integer, number of lines to be
        drawn between vmin and vmax. If given as array, values at which contour
        lines should be drawn. Set to 0 or [] to suppress drawing of contour lines
    contour_labels: boolean
        If True, add textual labels to the contour lines
    """
    x = np.linspace(w_2.min(), w_2.max(), density)
    y = np.linspace(w_c.min(), w_c.max(), density)
    z = griddata(w_2, w_c, val, x, y, interp='linear')
    if vmin is None:
        vmin=abs(z).min()
    if vmax is None:
        vmax=abs(z).max()
    if logscale:
        cmesh = ax_contour.pcolormesh(x, y, z, cmap=plt.cm.gnuplot2,
                                         norm=LogNorm(), vmax=vmax, vmin=vmin)
    else:
        if isinstance(contour_levels, int):
            levels = np.linspace(vmin, vmax, contour_levels)
        else:
            levels = contour_levels
        if len(levels) > 0:
            contour = ax_contour.contour(x, y, z, levels=levels,
                                         linewidths=0.5, colors='k')
            if contour_labels:
                ax_contour.clabel(contour, fontsize='smaller', lineine=1,
                                fmt='%g')
        cmesh = ax_contour.pcolormesh(x, y, z, cmap=plt.cm.gnuplot2,
                                      vmax=vmax, vmin=vmin)
    ax_contour.scatter(w_2, w_c, marker='o', c='cyan', s=5, zorder=10)
    ax_contour.set_xlabel(r"$\omega_2$ (GHz)")
    ax_contour.set_ylabel(r"$\omega_c$ (GHz)")
    fig = ax_cbar.figure
    cbar = fig.colorbar(cmesh, cax=ax_cbar)


def plot_C_loss(target_table, target='PE', loss_min=0.0, loss_max=1.0):
    """Plot concurrence and loss for all the categories in the given
    target_table.

    The target_table must contain the columns 'w1 [GHz]', 'w2 [GHz]',
    'wc [GHz]', 'C', 'loss', 'category', 'J_PE', 'J_SQ'

    It is assumed that the table contains results selected towards a specific
    target ('PE', or 'SQ'). That is, there must be a most one row for any tuple
    (w1, w2, wc, category). Specifically, the table returned by the
    get_stage2_table routine must be split before passing it to this routine.

    The 'target' parameter is only used to select for the final 'total' plot:
    if it is 'PE', rows with minimal value of 'J_PE' are selected, or minimal
    value of 'J_SQ' for 'SQ'.
    """
    plots = PlotGrid()
    table_grouped = target_table.groupby('category')
    for category in ['1freq_center', '1freq_random', '2freq_resonant',
    '2freq_random', '5freq_random', 'total']:
        if category == 'total':
            table = target_table\
                    .groupby(['w1 [GHz]', 'w2 [GHz]', 'wc [GHz]'],
                             as_index=False)\
                    .apply(lambda df: df.sort('J_%s'%target).head(1))\
                    .reset_index(level=0, drop=True)
        else:
            table = table_grouped.get_group(category)
        plots.add_cell(table['w2 [GHz]'], table['wc [GHz]'], table['C'],
                       vmin=0.0, vmax=1.0, contour_levels=11,
                       title='concurrence (%s_%s)'%(target, category))
        plots.add_cell(table['w2 [GHz]'], table['wc [GHz]'], table['loss'],
                       vmin=loss_min, vmax=loss_max,
                       contour_levels=11,
                       title='population loss (%s_%s)'%(target, category))
    plots.plot(quiet=True, show=True)


def plot_quality(t_PE, t_SQ):
    """Plot quality obtained from the two given tables.

    The tables t_PE and t_SQ must meet the requirements for the get_Q_table
    routine.
    """
    plots = PlotGrid()
    plots.n_cols = 2
    Q_table = get_Q_table(t_PE, t_SQ)
    table_grouped = Q_table.groupby('category')
    for category in ['1freq_center', '1freq_random', '2freq_resonant',
    '2freq_random', '5freq_random', 'total']:
        if category == 'total':
            table_grouped = Q_table.groupby(
            ['w1 [GHz]', 'w2 [GHz]', 'wc [GHz]'], as_index=False)
            table = Q_table\
                    .groupby(['w1 [GHz]', 'w2 [GHz]', 'wc [GHz]'],
                             as_index=False)\
                    .apply(lambda df: df.sort('Q').tail(1))\
                    .reset_index(level=0, drop=True)
        else:
            table = table_grouped.get_group(category)
        plots.add_cell(table['w2 [GHz]'], table['wc [GHz]'], table['Q'],
                       vmin=0.0, vmax=1.0, contour_levels=11,
                       title='quality (%s)'%(category,))
    plots.plot(quiet=True, show=True)


###############################################################################
#  Stage 1 Analysis                                                           #
###############################################################################


def stage1_rf_to_params(runfolder):
    """From the full path to a stage 1 runfolder, extract and return
    (w_2, w_c, E0, pulse_label), where w_2 is the second qubit frequency in
    MHz, w_c is the cavity frequency in MHz, E0 is the pulse amplitude in
    MHz, and pulse_label is an indicator of the pulse structure for that
    run, e.g. '2freq_resonant'

    Assumes that the runfolder path contains the structure

        w2_[w2]MHz_wc_[wc]MHz/stage1/[pulse label]

    If the pulse label is not 'field_free', then there must be a further
    subfolder "E[E0 in MHz]"
    """
    if runfolder.endswith(r'/'):
        runfolder = runfolder[:-1]
    E0 = None
    w_2 = None
    w_c = None
    pulse_label = None
    for part in runfolder.split(os.path.sep):
        if part == 'field_free':
            E0 = 0.0
            pulse_label = part
        E0_match = re.match("E(\d+)", part)
        if E0_match:
            E0 = int(E0_match.group(1))
        w2_wc_match = re.match(r'w2_(\d+)MHz_wc_(\d+)MHz', part)
        if w2_wc_match:
            w_2 = float(w2_wc_match.group(1))
            w_c = float(w2_wc_match.group(2))
        pulse_label_match = re.match(r'\dfreq_.*', part)
        if pulse_label_match:
            pulse_label = part
    if E0 is None:
        raise ValueError("Could not get E0 from %s" % runfolder)
    if w_2 is None:
        raise ValueError("Could not get w_2 from %s" % runfolder)
    if w_c is None:
        raise ValueError("Could not get w_c from %s" % runfolder)
    if pulse_label is None:
        raise ValueError("Could not get pulse_label from %s" % runfolder)
    return w_2, w_c, E0, pulse_label


def get_stage1_table(runs):
    """Summarize the results of the stage1 calculations in a DataFrame table.

    Assumes that the runfolder structure is

        [runs]/w2_[w2]MHz_wc_[wc]MHz/stage1/[pulse label]/E[E0 in MHz]

    except that the pulse amplitude subfolder is missing if the pulse label is
    'field free'.  Each runfolder must contain a file U.dat (resulting from
    propagation of pulse)

    The resulting table will have the columns

    'w1 [GHz]': value of left qubit frequency
    'w2 [GHz]': value of right qubit frequency
    'wc [GHz]': value of cavity frequency
    'C'       : Concurrence
    'loss',   : Loss from the logical subspace
    'E0 [MHz]': Peak amplitude of pulse
    'category': 'field_free', '1freq_center', '1freq_random', '2freq_resonant',
                '2freq_random', '5freq_random'
    'J_PE'    : Value of functional for perfect-entangler target
    'J_SQ'    : Value of functional for single-qubit target
    """
    runfolders = []
    for folder in find_folders(runs, 'stage1'):
        for subfolder in find_leaf_folders(folder):
            if os.path.isfile(os.path.join(subfolder, 'U.dat')):
                runfolders.append(subfolder)
    w1_s       = pd.Series(6.0, index=runfolders)
    w2_s       = pd.Series(index=runfolders)
    wc_s       = pd.Series(index=runfolders)
    C_s        = pd.Series(index=runfolders)
    loss_s     = pd.Series(index=runfolders)
    E0_s       = pd.Series(index=runfolders)
    category_s = pd.Series('', index=runfolders)
    for i, folder in enumerate(runfolders):
        w2, wc, E0, pulse_label = stage1_rf_to_params(folder)
        w2_s[i] = w2
        wc_s[i] = wc
        E0_s[i] = E0
        U_dat = os.path.join(folder, 'U.dat')
        U = QDYN.gate2q.Gate2Q(U_dat)
        C = U.closest_unitary().concurrence()
        loss = U.pop_loss()
        C_s[i] = C
        loss_s[i] = loss
        category_s[i] = re.sub('_\d+$', '_random', pulse_label)
    table = pd.DataFrame(OrderedDict([
                ('w1 [GHz]', w1_s),
                ('w2 [GHz]', w2_s/1000.0),
                ('wc [GHz]', wc_s/1000.0),
                ('C',        C_s),
                ('loss',     loss_s),
                ('E0 [MHz]', E0_s),
                ('category', category_s),
                ('J_PE',     1.0 - C_s + loss_s),
                ('J_SQ',     C_s + loss_s),
            ]))
    return table


###############################################################################
#  Stage 2 Analysis
###############################################################################

def get_stage2_table(runs):
    """Summarize the results of the stage2 calculations in a DataFrame table

    Assumes that the runfolder structure is
    [runs]/w2_[w2]MHz_wc_[wc]MHz/stage2/[target]_[category]/

    Each runfolder must contain a file U.dat (resulting from
    propagation of pulse_opt.json)

    The resulting table will have the columns

    'w1 [GHz]': value of left qubit frequency
    'w2 [GHz]': value of right qubit frequency
    'wc [GHz]': value of cavity frequency
    'C'       : Concurrence
    'loss',   : Loss from the logical subspace
    'category': 'field_free', '1freq_center', '1freq_random', '2freq_resonant',
                '2freq_random', '5freq_random'
    'target'  : 'PE', 'SQ'
    'J_PE'    : Value of functional for perfect-entangler target
    'J_SQ'    : Value of functional for single-qubit target
    """
    runfolders = []
    for folder in find_folders(runs, 'stage2'):
        for subfolder in find_leaf_folders(folder):
            if os.path.isfile(os.path.join(subfolder, 'U.dat')):
                runfolders.append(subfolder)
    w1_s       = pd.Series(6.0, index=runfolders)
    w2_s       = pd.Series(index=runfolders)
    wc_s       = pd.Series(index=runfolders)
    C_s        = pd.Series(index=runfolders)
    loss_s     = pd.Series(index=runfolders)
    category_s = pd.Series('', index=runfolders)
    target_s   = pd.Series('', index=runfolders)
    rx_folder = re.compile(r'''
                \/w2_(?P<w2>[\d.]+)MHz_wc_(?P<wc>[\d.]+)MHz
                \/stage2
                \/(?P<target>PE|SQ)
                  _(?P<category>1freq_center|1freq_random|2freq_resonant
                    |2freq_random|5freq_random)
                ''', re.X)
    for i, folder in enumerate(runfolders):
        m_folder = rx_folder.search(folder)
        if not m_folder:
            raise ValueError("%s does not match rx_folder" % folder)
        w2_s[i] = float(m_folder.group('w2'))
        wc_s[i] = float(m_folder.group('wc'))
        U_dat = os.path.join(folder, 'U.dat')
        U = QDYN.gate2q.Gate2Q(U_dat)
        C = U.closest_unitary().concurrence()
        loss = U.pop_loss()
        C_s[i] = C
        loss_s[i] = loss
        category_s[i] = m_folder.group('category')
        target_s[i] = m_folder.group('target')
    table = pd.DataFrame(OrderedDict([
                ('w1 [GHz]', w1_s),
                ('w2 [GHz]', w2_s/1000.0),
                ('wc [GHz]', wc_s/1000.0),
                ('C',        C_s),
                ('loss',     loss_s),
                ('category', category_s),
                ('target',   target_s),
                ('J_PE',     1.0 - C_s + loss_s),
                ('J_SQ',     C_s + loss_s),
            ]))
    return table


def get_Q_table(t_PE, t_SQ):
    """Combine two tables to calculate the overall "quality" function that
    expresses how well entanglement can both created and destroyed for a given
    parameter set.

    Arguments
    ---------

    t_PE: pandas.DataFrame
        Table containing (at least) columns 'w1 [GHz]', 'w2 [GHz]', 'wc [GHz]',
        'category', 'J_PE'

    t_SQ: pandas.DataFrame
        Table containing (at least) columns 'w1 [GHz]', 'w2 [GHz]', 'wc [GHz]',
        'category', 'J_SQ'

    Returns
    -------

    Q_table: pandas.DataFrame
        Table containing columns 'w1 [GHz]', 'w2 [GHz]', 'wc [GHz]',
        'category', 'J_PE', 'J_SQ', 'Q', where 'Q' is 1 - (J_PE + J_SQ)/2
    """
    Q_table = pd.concat(
        [t_PE[['w1 [GHz]', 'w2 [GHz]', 'wc [GHz]', 'category', 'J_PE']]
         .set_index(['w1 [GHz]', 'w2 [GHz]', 'wc [GHz]', 'category']),
         t_SQ[['w1 [GHz]', 'w2 [GHz]', 'wc [GHz]', 'category', 'J_SQ']]
         .set_index(['w1 [GHz]', 'w2 [GHz]', 'wc [GHz]', 'category'])
        ], axis=1).reset_index()
    Q_table['Q'] = 1 - 0.5*(Q_table['J_PE'] + Q_table['J_SQ'])
    return Q_table


###############################################################################
#  Workers                                                                    #
###############################################################################


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
    1: type = file, filename = pulse.guess, id = 1, check_tgrid = F, &
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


def avg_freq(analytical_pulse):
    """Given an analytical pulse, return the average frequency in GHz"""
    import numpy as np
    p = analytical_pulse.parameters
    if analytical_pulse.formula_name == 'field_free':
        return 6.0 # arbitrary (left qubit frequency)
    elif analytical_pulse.formula_name == '1freq':
        return p['w_L']
    elif analytical_pulse.formula_name == '2freq':
        s = abs(p['a_1']) + abs(p['a_2'])
        return (p['freq_1'] * abs(p['a_1']) + p['freq_2'] * abs(p['a_2']))/s
    elif analytical_pulse.formula_name == '5freq':
        weights = np.sqrt(np.abs(p['a_high'])**2 + np.abs(p['b_high'])**2)
        weights *= 1.0/np.sum(weights)
        return np.sum(weights * p['freq_high'])
    else:
        raise ValueError("Unknown formula name")


def max_freq_delta(analytical_pulse, w_L):
    """Return the maximum frequency that must be resolved for the given pulse
    in a rotating frame wL"""
    import numpy as np
    p = analytical_pulse.parameters
    if analytical_pulse.formula_name == 'field_free':
        return 0.0
    elif analytical_pulse.formula_name == '1freq':
        return abs(w_L - p['w_L'])
    elif analytical_pulse.formula_name == '2freq':
        return max(abs(w_L - p['freq_1']),  abs(w_L - p['freq_2']))
    elif analytical_pulse.formula_name == '5freq':
        return np.max(np.abs(w_L - p['freq_high']))
    else:
        raise ValueError("Unknown formula name")



def prop_RWA(config_file, pulse_json, outfolder, runfolder=None):
    """Given a config file and pulse file in the lab frame, modify them to be
    in the RWA and propagate. The propagation will take place in the given
    runfolder. If no runfolder is given, create a temporary runfolder which
    will be deleted after the propagation has finished. The file U.dat
    resulting from the propagation is copied to the given outfolder, as
    U_RWA.dat. Also, a file 'rwa_info.dat' is also written to the outfolder,
    detailing some of the parameters of the rotating frame. The outfolder may
    be identical to the runfolder.
    """
    import os
    import re
    from os.path import join
    from analytical_pulses import AnalyticalPulse
    from notebook_utils import avg_freq, max_freq_delta
    from clusterjob.utils import read_file, write_file
    from QDYN.shutil import mkdir, copy, rmtree
    import time
    import subprocess as sp
    import uuid

    p = AnalyticalPulse.read(pulse_json)
    config = read_file(config_file)

    rwa_info = ''
    w_d = avg_freq(p)
    rwa_info += "w_d = %f GHz\n" % w_d
    w_max = max_freq_delta(p, w_d)
    rwa_info += "max Delta = %f GHz\n" % w_max
    nt = max(1000, 100 * w_max * p.T) # 100 points per cycle
    p.nt = nt
    rwa_info += "nt = %d\n" % nt
    p._formula += '_rwa'
    p.parameters['w_d'] = w_d

    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '4'

    if runfolder is None:
        temp_runfolder = join(os.environ['SCRATCH_ROOT'], str(uuid.uuid4()))
    else:
        temp_runfolder = runfolder
    mkdir(temp_runfolder)
    mkdir(outfolder)
    try:
        config = re.sub('w_d\s*=\s*[\d.]+_MHz', 'w_d = %f_MHz' % (w_d*1000),
                        config)
        config = re.sub('nt\s*=\s*\d+', 'nt = %d'%nt,
                        config)
        config = re.sub(r'1: type = file, filename = pulse.guess, id = 1,\s*&',
                        r'1: type = file, filename = pulse.guess, id = 1, '
                        'check_tgrid = F, &', config)
        config = re.sub('is_complex\s*=\s*F', 'is_complex = T', config)

        write_file(join(temp_runfolder, 'config'), config)
        p.write(join(temp_runfolder, 'pulse.guess.json'), pretty=True)
        pulse = p.pulse(time_unit='ns', ampl_unit='MHz')
        pulse.write(join(temp_runfolder, 'pulse.guess'))
        start = time.time()
        with open(join(temp_runfolder, 'prop.log'), 'w', 0) as stdout:
            stdout.write("**** tm_en_gh --rwa --dissipation . \n")
            sp.call(['tm_en_gh', '--rwa', '--dissipation', '.'],
                    cwd=temp_runfolder, stderr=sp.STDOUT, stdout=stdout)
            stdout.write("**** rewrite_dissipation.py. \n")
            sp.call(['rewrite_dissipation.py',], cwd=temp_runfolder,
                    stderr=sp.STDOUT, stdout=stdout)
            stdout.write("**** tm_en_logical_eigenstates.py . \n")
            sp.call(['tm_en_logical_eigenstates.py', '.'],
                    cwd=temp_runfolder, stderr=sp.STDOUT, stdout=stdout)
            stdout.write("**** tm_en_prop . \n")
            sp.call(['tm_en_prop', '.'], cwd=temp_runfolder, env=env,
                    stderr=sp.STDOUT, stdout=stdout)
            end = time.time()
            stdout.write("**** finished in %s seconds . \n"%(end-start))
            rwa_info += "propagation time: %d seconds\n" % (end-start)
            copy(join(temp_runfolder, 'U.dat'), join(outfolder, 'U_RWA.dat'))
            copy(join(temp_runfolder, 'prop.log'),
                 join(outfolder, 'prop_rwa.dat'))
        write_file(join(outfolder, 'rwa_info.dat'), rwa_info)
    except Exception as e:
        print e
    finally:
        if runfolder is None:
            rmtree(temp_runfolder)


def compare_RWA_prop(runfolder_original, run_root, use_pulse='pulse_opt.json'):
    """
    Take the file 'config' and the file given by `use_pulse` inside the given
    `runfolder_original`. Assume these files are in the lab frame, and
    (re-)propagate them both in the lab and in the RWA frame. The runfolders
    for these new propagations are subfolders for run_root, 'LAB' for the lab
    frame, and 'RWA' for the RWA frame.
    """
    from os.path import join
    from analytical_pulses import AnalyticalPulse
    import QDYN
    from QDYN.shutil import mkdir, copy
    import time
    import subprocess as sp
    p = AnalyticalPulse.read(
        join(runfolder_original, use_pulse))
    pulse = p.pulse(time_unit='ns', ampl_unit='MHz')
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '4'
    lab_runfolder = join(run_root, 'LAB')
    rwa_runfolder = join(run_root, 'RWA')
    mkdir(lab_runfolder)
    mkdir(rwa_runfolder)

    # re-propagate the original frame
    def worker(mode):
        if mode == 'LAB':
            copy(join(runfolder_original, 'config'), lab_runfolder)
            pulse.write(join(lab_runfolder, 'pulse.guess'))
            start = time.time()
            with open(os.path.join(lab_runfolder, 'prop.log'), 'w', 0) \
            as stdout:
                stdout.write("**** tm_en_gh --dissipation . \n")
                sp.call(['tm_en_gh', '--dissipation', '.'], cwd=lab_runfolder,
                        stderr=sp.STDOUT, stdout=stdout)
                stdout.write("**** rewrite_dissipation.py. \n")
                sp.call(['rewrite_dissipation.py',], cwd=lab_runfolder,
                        stderr=sp.STDOUT, stdout=stdout)
                stdout.write("**** tm_en_logical_eigenstates.py . \n")
                sp.call(['tm_en_logical_eigenstates.py', '.'],
                        cwd=lab_runfolder, stderr=sp.STDOUT, stdout=stdout)
                stdout.write("**** tm_en_prop . \n")
                sp.call(['tm_en_prop', '.'], cwd=lab_runfolder, env=env,
                        stderr=sp.STDOUT, stdout=stdout)
                end = time.time()
                stdout.write("**** finished in %s seconds . \n"%(end-start))
        elif mode == 'RWA':
            # re-propagate the RWA
            prop_RWA(join(runfolder_original, 'config'),
                     join(runfolder_original, use_pulse),
                     outfolder=rwa_runfolder, runfolder=rwa_runfolder)
        else:
            raise ValueError("Invalide mode: %s" % mode)

    worker('RWA')
    worker('LAB')

    U_LAB = QDYN.gate2q.Gate2Q(join(lab_runfolder, 'U.dat'))
    U_RWA = QDYN.gate2q.Gate2Q(join(rwa_runfolder, 'U.dat'))
    return U_LAB, U_RWA


def get_RWA_table(runs):
    """Summarize the results of the RWA propagation

    Looks for U_RWA.dat in all the subfolders of the given `runs` folder

    The resulting table will have the columns

    'C (RWA)'              : Concurrence
    'loss (RWA)',          : Loss from the logical subspace
    'nt (RWA)',            : Numer of time steps used
    'wd [GHz]'             : Frequency of rotating frame
    'max Delta (RWA) [GHz]': Max. required frequency in rotating frame
    'prop time (RWA) [s]'  : Seconds required for propagation

    and use the runfolder name as the index
    """
    runfolders = []
    for U_RWA_dat in find_files(runs, 'U_RWA.dat'):
        runfolders.append(os.path.split(U_RWA_dat)[0])
    C_s         = pd.Series(index=runfolders)
    loss_s      = pd.Series(index=runfolders)
    nt_s        = pd.Series(index=runfolders, dtype=np.int)
    wd_s        = pd.Series(index=runfolders)
    max_delta_s = pd.Series(index=runfolders)
    prop_time_s = pd.Series(index=runfolders)
    i = 0
    for i, folder in enumerate(runfolders):
        U_dat = os.path.join(folder, 'U_RWA.dat')
        U = QDYN.gate2q.Gate2Q(U_dat)
        try:
            C = U.closest_unitary().concurrence()
            loss = U.pop_loss()
            C_s[i] = C
            loss_s[i] = loss
        except ValueError:
            print "%s is invalid" % U_dat
        info_dat = os.path.join(folder, 'rwa_info.dat')
        if not os.path.isfile(info_dat):
            continue
        with open(info_dat) as info:
            for line in info:
                m = re.match('w_d\s*=\s*([\d.]+)\s*GHz', line)
                if m:
                    wd_s[i] = float(m.group(1))
                m = re.match('max Delta\s*=\s*([\d.]+)\s*GHz', line)
                if m:
                    max_delta_s[i] = float(m.group(1))
                m = re.match('nt\s*=\s*(\d+)', line)
                if m:
                    nt_s[i] = int(m.group(1))
                m = re.match('propagation time:\s*(\d+)\s*seconds', line)
                if m:
                    prop_time_s[i] = float(m.group(1))
    table = pd.DataFrame(OrderedDict([
                ('C (RWA)',               C_s),
                ('loss (RWA)',            loss_s),
                ('nt (RWA)',              nt_s),
                ('wd (RWA) [GHz]',        wd_s),
                ('max Delta (RWA) [GHz]', max_delta_s),
                ('prop time (RWA) [s]',   prop_time_s),
            ]))
    return table
