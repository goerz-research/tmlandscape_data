"""
Module containing auxilliary routines for the notebooks in this folder
"""
import QDYN
import re
import os
import StringIO
import urllib
import base64
import numpy as np
from matplotlib.mlab import griddata
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm, ColorConverter
from collections import OrderedDict
from textwrap import dedent
from IPython.display import display, HTML, Markdown, Latex, Math
from mgplottools.mpl import set_axis, new_figure
from matplotlib import rcParams
from matplotlib.patches import Rectangle
from analytical_pulses import AnalyticalPulse
from QDYN.pulse import tgrid_from_config
from QDYN.shutil import copy
from QDYN.weyl import WeylChamber
import QDYNTransmonLib
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
    cell_height: float
        height of each panel area [cm]
    left_margin: float
        distance from left cell edge to axes [cm]
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
    cbar_title: boolean
        If True, show titles along the colorbar. If False, show title on top
        of the axes.
    wc_min: float
        Minimum cavity frequency [GHz].
    wc_max: float
        Maximum cavity frequency [GHz]
    w2_min: float
        Minimum qubit frequency [GHz].
    w2_max: float
        Maximum qubit frequency [GHz].
    draw_cell_box: boolean
        Draw a box around each cell (for debugging, defaults to False)
    """
    def __init__(self, layout='inline'):
        if layout == 'poster':
            self.cell_width      =  6.5
            self.cell_height     =  6.2
            self.left_margin     =  1.0
            self.bottom_margin   =  0.7
            self.h               =  5.3
            self.w               =  4.0
            self.cbar_width      =  0.2
            self.cbar_gap        =  0.5
            self.density         =  100
            self.n_cols          =  2
            self.contour_labels  = False
            self.cbar_title      = True
            self.ylabelpad       = -1.0
            self.xlabelpad       =  0.3
            self.clabelpad       =  1.0
            self.scatter_size    =  0.0
        elif layout == 'paper':
            self.cell_width      =  6.5
            self.cell_height     =  6.2
            self.left_margin     =  1.0
            self.bottom_margin   =  0.7
            self.h               =  5.3
            self.w               =  4.0
            self.cbar_width      =  0.2
            self.cbar_gap        =  0.5
            self.density         =  100
            self.n_cols          =  2
            self.contour_labels  = False
            self.cbar_title      = True
            self.ylabelpad       = -1.0
            self.xlabelpad       =  0.3
            self.clabelpad       =  1.0
            self.scatter_size    =  0.0
        elif layout == 'inline':
            self.cell_width      =  15.0
            self.cell_height     =  10.0 + 1.25 + 0.8
            self.left_margin     =  1.8
            self.bottom_margin   =  1.25
            self.h               =  10.0
            self.w               =  10.0
            self.cbar_width      =  0.3
            self.cbar_gap        =  0.5
            self.density         =  100
            self.n_cols          =  2
            self.contour_labels  = False
            self.cbar_title      = False
            self.ylabelpad       =  1.0
            self.xlabelpad       =  1.0
            self.clabelpad       =  1.0
            self.scatter_size    =  5.0
        else:
            raise ValueError("Unknown 'layout' value")
        self._cells          = [] # array of cell_dicts
        self.wc_min = 4.5
        self.wc_max = 11.1
        self.w2_min = 5.0
        self.w2_max = 7.5
        self.x_major_ticks = 0.5
        self.x_minor = 5
        self.y_major_ticks = 0.5
        self.y_minor = 5
        self.draw_cell_box = False
        self._axes = []
        # after rendering, self._axes[i] is a list of axes in the i'th cell
        self._cbars = []
        # after rendering, self._cbars[i] is the colorbar in the i'th cell

    def add_cell(self, w2, wc, val, val_alpha=None, logscale=False, vmin=None,
            vmax=None, contour_levels=0, title=None, cmap=None, bg='white',
            x_labels=True, y_labels=True, left_margin=None,
            bottom_margin=None):
        """Add a cell to the plot grid

        The `left_margin` and `bottom_margin` parameters can be used to
        override the corresponding instance attribute for that specific cell
        (in order to account for for changes in the labeling.

        All other parameters will be passed to the render_values routine
        """
        cell_dict = {}
        cell_dict['w2'] = w2
        cell_dict['wc'] = wc
        cell_dict['val'] = val
        cell_dict['val_alpha'] = val_alpha
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
        cell_dict['cmap'] = cmap
        cell_dict['bg'] = bg
        cell_dict['x_labels'] = x_labels
        cell_dict['y_labels'] = y_labels
        if left_margin is not None:
            cell_dict['left_margin'] = left_margin
        if bottom_margin is not None:
            cell_dict['bottom_margin'] = bottom_margin
        self._cells.append(cell_dict)

    def plot(self, quiet=True, show=True, style=None):

        n_cells = len(self._cells)
        assert n_cells > 0, "No cells to plot"
        fig_width = self.n_cols * self.cell_width
        n_cols = self.n_cols
        n_rows = n_cells // n_cols
        if n_rows * n_cols < n_cells:
            n_rows += 1
        cell_width = self.cell_width
        cell_height = self.cell_height
        fig_height = n_rows * cell_height
        fig = new_figure(fig_width, fig_height, quiet=quiet, style=style)

        col = 0
        row = 0

        self._axes = []
        self._cbars = []

        if self.draw_cell_box:
            # insert hidden axes instance
            ax_background = fig.add_axes([0,0,1,1], frameon=False)
            ax_background.axes.get_yaxis().set_visible(False)
            ax_background.axes.get_xaxis().set_visible(False)

        for i, cell_dict in enumerate(self._cells):

            cell_axes = []

            # draw a rectangle around the cell
            if self.draw_cell_box:
                ax_background.add_patch(
                    Rectangle(
                        ( (col*cell_width)/fig_width,
                        (n_rows-row-1)*cell_height/fig_height
                        ), cell_width/fig_width, cell_height/fig_height,
                        edgecolor='black', fill=False
                    )
                )

            left_margin = self.left_margin
            if 'left_margin' in cell_dict:
                left_margin = cell_dict['left_margin']
            bottom_margin = self.bottom_margin
            if 'bottom_margin' in cell_dict:
                bottom_margin = cell_dict['bottom_margin']
            pos_contour = [(col*cell_width + left_margin)/fig_width,
                           ((n_rows-row-1)*cell_height
                            + bottom_margin)/fig_height,
                            self.w/fig_width, self.h/fig_height]
            ax_contour = fig.add_axes(pos_contour)
            cell_axes.append(ax_contour)

            pos_cbar = [(col*cell_width + left_margin + self.w
                         + self.cbar_gap)/fig_width,
                        ((n_rows-row-1)*cell_height
                            + bottom_margin)/fig_height,
                            self.cbar_width/fig_width, self.h/fig_height]
            ax_cbar = fig.add_axes(pos_cbar)
            cell_axes.append(ax_cbar)

            if self.cbar_title:
                axT = ax_cbar.twinx()
                cell_axes.append(axT)
                cbar = render_values(cell_dict['w2'], cell_dict['wc'],
                              cell_dict['val'], ax_contour, axT,
                              density=self.density,
                              logscale=cell_dict['logscale'],
                              val_alpha=cell_dict['val_alpha'],
                              vmin=cell_dict['vmin'], vmax=cell_dict['vmax'],
                              contour_levels=cell_dict['contour_levels'],
                              contour_labels=self.contour_labels,
                              scatter_size=self.scatter_size,
                              cmap=cell_dict['cmap'], bg=cell_dict['bg'],
                              x_labels=cell_dict['x_labels'],
                              y_labels=cell_dict['y_labels'])
                ax_cbar.set_yticks([])
                ax_cbar.set_yticklabels('',visible=False)
            else:
                cbar = render_values(cell_dict['w2'], cell_dict['wc'],
                              cell_dict['val'], ax_contour, ax_cbar,
                              density=self.density,
                              val_alpha=cell_dict['val_alpha'],
                              logscale=cell_dict['logscale'],
                              vmin=cell_dict['vmin'], vmax=cell_dict['vmax'],
                              contour_levels=cell_dict['contour_levels'],
                              contour_labels=self.contour_labels,
                              scatter_size=self.scatter_size,
                              cmap=cell_dict['cmap'], bg=cell_dict['bg'],
                              x_labels=cell_dict['x_labels'],
                              y_labels=cell_dict['y_labels'])

            # show the resonance line (cavity on resonace with qubit 2)
            ax_contour.plot(np.linspace(self.wc_min, self.wc_max, 10),
                            np.linspace(self.wc_min, self.wc_max, 10),
                            color='white')
            ax_contour.plot(np.linspace(self.w2_min, self.w2_max, 10),
                            6.0*np.ones(10), color='white')
            ax_contour.axvline(6.29, color='white', ls='--')
            ax_contour.axvline(6.31, color='white', ls='--')
            ax_contour.axvline(5.71, color='white', ls='--')
            ax_contour.axvline(6.0,  color='white', ls='-')
            ax_contour.axvline(5.69, color='white', ls='--')
            #ax_contour.axvline(6.58, color='white', ls='--')
            #ax_contour.axvline(6.62, color='white', ls='--')
            # ticks and axis labels
            set_axis(ax_contour, 'x', self.w2_min, self.w2_max,
                     self.x_major_ticks, minor=self.x_minor)
            set_axis(ax_contour, 'y', self.wc_min, self.wc_max,
                     self.y_major_ticks, minor=self.y_minor)
            ax_contour.tick_params(which='both', direction='out')
            if cell_dict['x_labels']:
                ax_contour.set_xlabel(r"$\omega_2$ (GHz)",
                                      labelpad=self.xlabelpad)
            else:
                ax_contour.set_xticklabels([])
            if cell_dict['y_labels']:
                ax_contour.set_ylabel(r"$\omega_c$ (GHz)",
                                      labelpad=self.ylabelpad)
            else:
                ax_contour.set_yticklabels([])

            if cell_dict['title'] is not None:
                if self.cbar_title:
                    ax_cbar.set_ylabel(cell_dict['title'],
                                       labelpad=self.clabelpad)
                else:
                    ax_contour.set_title(cell_dict['title'])

            col += 1
            if col == self.n_cols:
                col = 0
                row += 1

            self._axes.append(cell_axes)
            self._cbars.append(cbar)

        if show:
            plt.show(fig)
            plt.close(fig)
        else:
            return fig


def get_w_d_from_config(config_file):
    """Return driving frequency from the given config file, in GHz

    Raise AssertionError is config file does not contain w_d
    """
    config_w_d = None
    with open(config_file) as in_fh:
        for line in in_fh:
            m = re.match(r'w_d\s*=\s*([\d.]+)_MHz', line)
            if m:
                config_w_d = float(m.group(1))
    assert config_w_d is not None, "config does not contain w_d"
    return config_w_d / 1000.0


def pulse_config_compat(analytical_pulse, config_file, adapt_config=False):
    """Ensure that the given config file matches the time grid and RWA
    parameters for the given analytical pulse, or raise an AssertionError
    otherwise. If adapt_config is True, instead of raising an error, try to
    rewrite the config file to match the pulse.
    """
    try:
        # time grid
        config_tgrid, time_unit = tgrid_from_config(config_file,
                                                    pulse_grid=False)
        assert time_unit == analytical_pulse.time_unit, \
            "time_unit does not match"
        assert len(config_tgrid) == analytical_pulse.nt, \
            "nt does not match"
        assert (abs(config_tgrid[-1] - analytical_pulse.T) < 1.0e-8), \
            "T does not match"
        # RWA parameters
        if 'rwa' in analytical_pulse.formula_name:
            config_w_d = get_w_d_from_config(config_file)
            assert 'w_d' in analytical_pulse.parameters
            pulse_w_d = analytical_pulse.parameters['w_d'] # GHz
            assert (abs(config_w_d - pulse_w_d) < 1.0e-8), 'w_d does not match'
    except AssertionError:
        if adapt_config:
            copy(config_file, config_file + '~')
            T = analytical_pulse.T
            t0 = analytical_pulse.t0
            assert t0 == 0.0, "t0 is not 0.0"
            nt = analytical_pulse.nt
            unit = analytical_pulse.time_unit
            with open(config_file+'~') as in_fh, \
            open(config_file, 'w') as out_fh:
                for line in in_fh:
                    line = re.sub(r't_start\s*=\s*[\d.]+(_\w+)?',
                                 't_start = 0.0', line)
                    line = re.sub(r't_stop\s*=\s*[\d.]+(_\w+)?',
                                  't_stop = %.1f_%s' % (T, unit), line)
                    line = re.sub(r'nt\s*=\s*\d+', 'nt = %d' % nt, line)
                    if 'rwa' in analytical_pulse.formula_name:
                        if 'w_d' in line:
                            pulse_w_d = analytical_pulse.parameters['w_d'] \
                                        * 1000.0 # MHz
                            line = re.sub(r'w_d\s*=\s*([\d.]+)_MHz',
                                          r'w_d = %f_MHz' % pulse_w_d, line)
                    out_fh.write(line)
        else:
            raise
    return True


def render_values(w_2, w_c, val, ax_contour, ax_cbar, density=100,
    logscale=False, val_alpha=None, vmin=None, vmax=None, contour_levels=11,
    contour_labels=False, scatter_size=5, clip=True, cmap=None, bg='white',
    y_labels=True, x_labels=True):
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
    density: int
        Number of points per GHz to be used for interpolating the plots
    val_alpha: array or None
        Array of alpha (transparency) values, with each value in [0, 1]
    vmin: float
        Bottom value of the z-axis (colorbar range). Defaults to the minimum
        value in the val array
    vmax: float
        Top value of the z-axis (colorbar range). Defaults to the maximum
        value in the val array
    contour_levels: int, array of floats
        Contour lines to draw. If given as an integer, number of lines to be
        drawn between vmin and vmax. If given as array, values at which contour
        lines should be drawn. Set to 0 or [] to suppress drawing of contour
        lines
    contour_labels: boolean
        If True, add textual labels to the contour lines
    scatter_size: float
        Size of the scatter points. Set to zero to disable showing points
    clip: boolean
        If True, clip val to the range [vmin:vmax]
    cmap: matplotlib.colors.Colormap instance, or None
        Colormap to be used. If None, colormap will be chosen automatically
        based on the data.
    bg: string
        Color of background, i.e., the color that val_alpha will fade into
    x_labels: boolean
        If False, suppress both the x-ticklabels and the x-axis-label
    y_labels: boolean
        If False, suppress both the y-ticklabels and the y-axis-label

    Returns
    -------

    cbar instance
    """
    x = np.linspace(w_2.min(), w_2.max(), int((w_2.max()-w_2.min())*density))
    y = np.linspace(w_c.min(), w_c.max(), int((w_c.max()-w_c.min())*density))
    z = griddata(w_2, w_c, val, x, y, interp='linear')
    z_alpha = None
    if val_alpha is not None:
        z_alpha = griddata(w_2, w_c, val_alpha, x, y, interp='linear')
    if clip:
        np.clip(z, vmin, vmax, out=z)
    if vmin is None:
        vmin=abs(z).min()
    if vmax is None:
        vmax=abs(z).max()
    if logscale:
        if cmap is None:
            cmap=plt.cm.gnuplot2
        normalize = LogNorm()
    else: # linear scale
        if cmap is None:
            if vmin < 0.0 and vmax > 0.0:
                # divergent colormap
                cmap = plt.cm.RdYlBu_r
            else:
                # sequential colormap
                cmap = plt.cm.gnuplot2
        normalize = plt.Normalize(vmin, vmax)
    cmesh = ax_contour.pcolormesh(x, y, z, cmap=cmap, visible=False,
                                  norm=normalize, vmax=vmax, vmin=vmin)
    rgba = cmap(normalize(z))
    if z_alpha is not None:
        color_convert = ColorConverter()
        bg_rgb = color_convert.to_rgb(bg)
        for i in [0, 1, 2]:
            rgba[:,:,i] *= z_alpha[:,:]
            rgba[:,:,i] += (1.0-z_alpha[:,:]) * bg_rgb[i]
    ax_contour.imshow(rgba, interpolation='nearest', aspect='auto',
                        extent=[w_2.min(),w_2.max(),w_c.min(),w_c.max()],
                        origin='lower')
    if not logscale:
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
    if scatter_size > 0:
        ax_contour.scatter(w_2, w_c, marker='o', c='cyan', s=scatter_size,
                        linewidth=0.1*scatter_size, zorder=10)
    ax_contour.set_axis_bgcolor('white')
    fig = ax_cbar.figure
    cbar = fig.colorbar(cmesh, cax=ax_cbar)
    return cbar


def plot_C_loss(target_table, target='PE', C_min=0.0, C_max=1.0,
    loss_min=0.0, loss_max=1.0, outfile=None, include_total=True,
    categories=None, show_oct_improvement=False, scale=1.0, logscale=False,
    concurrence_error=False, scatter_size=0):
    """Plot concurrence and loss for all the categories in the given
    target_table.

    The target_table must contain the columns 'w1 [GHz]', 'w2 [GHz]',
    'wc [GHz]', 'C', 'max loss', 'category', 'J_PE', 'J_SQ'

    If show_oct_improvement is True, is must also contain the columns
    'C (guess)', 'max loss (guess)'

    It is assumed that the table contains results selected towards a specific
    target ('PE', or 'SQ'). That is, there must be a most one row for any tuple
    (w1, w2, wc, category). Specifically, the table returned by the
    get_stage2_table routine must be split before passing it to this routine.

    The 'target' parameter is only used to select for the final 'total' plot:
    if it is 'PE', rows with minimal value of 'J_PE' are selected, or minimal
    value of 'J_SQ' for 'SQ'.

    If outfile is given, write to outfile instead of showing plot.

    If include_total is True, show the best values from any category in a panel

    By default, the list of categories are those used in stage2 (simplex
    optimization). The list of categories may be overrriden by passing it to
    this routine.

    If show_oct_imrpovment is True, show the improvement made by OCT are shown
    instead of the actual values.

    The scale factor scales all shown values. If given, it will usually take
    the value -1

    If concurrence_error is True, show 1-C instead of C (ignored for
    show_oct_improvement)
    """
    plots = PlotGrid()
    if scatter_size is not None:
        plots.scatter_size = scatter_size
    table_grouped = target_table.groupby('category')
    if categories is None:
        categories = ['1freq_center', '1freq_random', '2freq_resonant',
        '2freq_random', '5freq_random']
    if include_total:
        if not show_oct_improvement:
            categories.append('total')
    bg = 'white'
    for category in categories:
        try:
            if category == 'total':
                table = target_table\
                        .groupby(['w1 [GHz]', 'w2 [GHz]', 'wc [GHz]'],
                                as_index=False)\
                        .apply(lambda df: df.sort('J_%s'%target).head(1))\
                        .reset_index(level=0, drop=True)
            else:
                table = table_grouped.get_group(category)
        except KeyError:
            continue
        if show_oct_improvement:
            if target == 'SQ':
                title = 'decrease C (%s_%s)'%(target, category)
                if scale != 1.0:
                    title += ' scaled by %g' % scale
                plots.add_cell(table['w2 [GHz]'], table['wc [GHz]'],
                        scale*(table['C (guess)']-table['C']),
                        val_alpha=(1-table['max loss']), bg=bg,
                        vmin=C_min, vmax=C_max, contour_levels=0,
                        logscale=logscale, title=title)
            else:
                title = 'increase C (%s_%s)'%(target, category)
                if scale != 1.0:
                    title += ' scaled by %g' % scale
                plots.add_cell(table['w2 [GHz]'], table['wc [GHz]'],
                        scale*(table['C']-table['C (guess)']),
                        val_alpha=(1-table['max loss']), bg=bg,
                        vmin=C_min, vmax=C_max, contour_levels=0,
                        logscale=logscale, title=title)
            title = 'decrease pop loss (%s_%s)'%(target, category)
            if scale != 1.0:
                title += ' scaled by %g' % scale
            plots.add_cell(table['w2 [GHz]'], table['wc [GHz]'],
                       scale*(table['max loss (guess)']-table['max loss']),
                       logscale=logscale, vmin=loss_min, vmax=loss_max,
                       bg=bg, contour_levels=0, title=title)
        else:
            if concurrence_error:
                title = 'concurrence error (%s_%s)'%(target, category)
            else:
                title = 'concurrence (%s_%s)'%(target, category)
                bg = 'black'
            if scale != 0.0:
                title += ' scaled by %g' % scale
            if concurrence_error:
                plots.add_cell(table['w2 [GHz]'], table['wc [GHz]'],
                            scale*(1-table['C']),
                            val_alpha=(1-table['max loss']),
                            vmin=C_min, vmax=C_max, bg=bg,
                            contour_levels=0, logscale=logscale, title=title)
            else:
                plots.add_cell(table['w2 [GHz]'], table['wc [GHz]'],
                            scale*table['C'], val_alpha=(1-table['max loss']),
                            vmin=C_min, vmax=C_max, bg=bg,
                            contour_levels=0, logscale=logscale, title=title)
            title='population loss (%s_%s)'%(target, category)
            if scale != 1.0:
                title += ' scaled by %g' % scale
            plots.add_cell(table['w2 [GHz]'], table['wc [GHz]'],
                        scale*table['max loss'], vmin=loss_min, vmax=loss_max,
                        contour_levels=0, logscale=logscale, title=title)

    if outfile is None:
        plots.plot(quiet=True, show=True)
    else:
        fig = plots.plot(quiet=True, show=False)
        fig.savefig(outfile)
        plt.close(fig)


def filter_table(table, select_points):
    """Given a table that must contain the columns 'w2 [GHz]' and 'wc [GHz]',
    and a list of points (w2, wc), return a copy of the table with only the
    rows exactly matching the values for w2 and wc
    """
    w2, wc = select_points[0]
    filter = (table['w2 [GHz]'] == w2) & (table['wc [GHz]'] == wc)
    for (w2, wc) in select_points[1:]:
        filter = filter | (table['w2 [GHz]'] == w2) & (table['wc [GHz]'] == wc)
    return table[filter]


def latex_float(f):
    """Format a float for LaTeX, using scientific notation"""
    float_str = "{0:.2e}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"${0} \times 10^{{{1}}}$".format(base, int(exponent))
    else:
        return float_str


def fig_to_html(fig):
    """Convert a matplotlib figures into into an inline HTML snippet that
    displays the figure"""
    imgdata = StringIO.StringIO()
    fig.savefig(imgdata, format='png', dpi=70)
    imgdata.seek(0)
    uri = 'data:image/png;base64,' + urllib.quote(base64.b64encode(imgdata.buf))
    return '<img src = "%s"/>' % uri


def pd_insert_col_width(html_str, widths):
    """Modify the html string representing a pandas DataFrame in HTML
    (df.to_html) to ensure that each column has the specified width in pixels
    """
    lines = ['<colgroup>']
    for width in widths:
        lines.append('<col span="1" style="width: %dpx;">' % int(width))
    lines.append('</colgroup>')
    lines.append('<thead>')
    width = int(np.sum(np.array(widths)))
    html_str = html_str.replace('border=', 'width="%spx" border='%width)
    return html_str.replace('<thead>', "\n".join(lines))


def get_max_1freq_quality(t_PE, t_SQ):
    """Return the maximum quality value for a single frequency pulse, based on
    the given input tables"""
    Q_table = get_Q_table(t_PE, t_SQ)
    Q_table = Q_table[Q_table['category'].str.contains('1freq')]
    table_grouped = Q_table.groupby(
    ['w1 [GHz]', 'w2 [GHz]', 'wc [GHz]'], as_index=False)
    Q_table = Q_table\
              .groupby(['w1 [GHz]', 'w2 [GHz]', 'wc [GHz]'],
                      as_index=False)\
              .apply(lambda df: df.sort('Q').tail(1))\
              .reset_index(level=0, drop=True)
    return Q_table['Q'].max()


def plot_zeta_data(zeta_table, T, scatter_size=0, outfile=None):
    plots = PlotGrid()
    plots.scatter_size = scatter_size
    zeta = zeta_table['zeta [MHz]']
    abs_zeta = np.clip(np.abs(zeta), a_min=1e-5, a_max=1e5)
    w2 = zeta_table['w2 [GHz]']
    wc = zeta_table['wc [GHz]']

    plots.add_cell(w2, wc, abs_zeta, title=r'$\zeta$ (MHz)', logscale=True,
                   vmin=1e-3, left_margin=1.2, y_labels=True)

    T_entangling = 500.0/abs_zeta
    plots.add_cell(w2, wc, T_entangling, logscale=True, vmax=1e3,
                    #vmin=0.0, vmax=100.0,
                   title='$T(\gamma=\pi)$ (ns)', left_margin=0.7, y_labels=False)

    gamma = -2.0 * np.pi * (zeta/1000.0) * T # entangling phase
    C = np.abs(np.sin(0.5*gamma))
    plots.add_cell(w2, wc, C, vmin=0.0, vmax=1.0,
                   title='concurrence at $T = %s$ ns' % T,
                   left_margin=1.2, y_labels=True)

    gamma_bare = 0.012
    rel_decay = zeta_table['gamma [MHz]'] / gamma_bare
    plots.add_cell(w2, wc, rel_decay, logscale=False, vmin=1, vmax=2.3,
                   title=r'$\gamma_{dressed} / \gamma_{bare}$',
                   left_margin=0.7, y_labels=False)
    if outfile is None:
        plots.plot(quiet=True, show=True)
    else:
        fig = plots.plot(quiet=True, show=False)
        fig.savefig(outfile)
        plt.close(fig)


def plot_quality(t_PE, t_SQ, outfile=None, include_total=True,
    categories=None, vmin=1.0e-3, vmax=1.0, scatter_size=0):
    """Plot quality obtained from the two given tables.

    The tables t_PE and t_SQ must meet the requirements for the get_Q_table
    routine.

    If outfile is given, write to outfile instead of showing plot.

    If not given, the categories default to those used in stage2
    """
    plots = PlotGrid()
    if scatter_size is not None:
        plots.scatter_size = scatter_size
    plots.n_cols = 2
    Q_table = get_Q_table(t_PE, t_SQ)
    table_grouped = Q_table.groupby('category')
    if categories is None:
        categories = ['1freq_center', '1freq_random', '2freq_resonant',
        '2freq_random', '5freq_random']
    if include_total:
        categories.append('total')
    for category in categories:
        try:
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
        except KeyError:
            continue
        plots.add_cell(table['w2 [GHz]'], table['wc [GHz]'], 1.0-table['Q'],
                       vmin=vmin, vmax=vmax, logscale=True,
                       title='1-quality (%s)'%(category,))
    if outfile is None:
        plots.plot(quiet=True, show=True)
    else:
        fig = plots.plot(quiet=True, show=False)
        fig.savefig(outfile)
        plt.close(fig)


def diss_error(gamma, t):
    """Return the average loss of population for a field-free propagation of
    the bare qubit states, with decay rate gamma in GHz and the gate duration
    in ns. This is the lower limit for the reachable gate error, due to decay
    of the qubit state"""
    two_pi = 2.0 * np.pi
    # note that *amplitudes* decay at half the rate of the populations
    U = QDYN.gate2q.Gate2Q(np.diag([
        1.0,
        np.exp(-0.5*two_pi * gamma * t),
        np.exp(-0.5*two_pi * gamma * t),
        np.exp(-0.5*two_pi * (2*gamma) * t)]))
    O = QDYN.gate2q.identity
    # U is the time evolution operator in the field-free case for the bare
    # qubit system. Mixing with the cavity due to qubit-cavity interaction is
    # ignored and will need to additional error. Therefor, diss_error is still
    # a lower (global) limit
    #return 1.0 - 0.25*(U.conjugate().dot(U)).trace()
    return 1.0 - U.F_avg(QDYN.gate2q.identity)


###############################################################################
#  Stage 1 Analysis                                                           #
###############################################################################


def level_to_qu_numbers(level_index, n_qubit, n_cavity):
    """Convert 1-based level index to quantum number (i,j,n)"""
    l = level_index - 1
    nn = n_qubit * n_cavity
    i = l // nn; l = l - i * nn
    j = l // n_cavity
    n = l - j * n_cavity
    return (i,j,n)


def bare_decomposition(state, n_qubit, n_cavity, limit=0.001):
    """Return the decomposition into the dominant bare states.

    Arguments:
        state (numpy array): Complex amplitude array of a quantum state
        EV (QDYNTransmonLib.io.fullHamLevels): Level and eigenvalue information
    """
    label = lambda k: "{%d%d%d}" % level_to_qu_numbers(k+1, n_qubit, n_cavity)
    result = ""
    for k in range(len(state)):
        i, j, n = level_to_qu_numbers(k+1, n_qubit, n_cavity)
        if abs(state[k])**2 > limit:
            result += " + {percent:.1f}% {label}".format(
                          percent=(100*abs(state[k])**2),
                          label=label(k))
    return result[3:] # strip out leading ' + '


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


def J_target(target, concurrence, loss):
    """Figure of meriit, for PE or SQ target"""
    if target == 'PE':
        return J_PE(concurrence, loss)
    elif target == 'SQ':
        return J_SQ(concurrence, loss)
    else:
        raise ValueError("target must be PE, SQ")


def J_PE(concurrence, loss):
    """Perfect entangler figure of merit"""
    return 1.0 - concurrence + concurrence*loss


def J_SQ(concurrence, loss):
    """Single qubit figure of merit"""
    return loss + concurrence - concurrence*loss


def get_zeta_table(runs, T=None, limit=1e-3):
    """Summarize the value of the field-free entangling energy zeta over the
    parameter space

    Assumes that the runfolder structure is

        [runs]/w2_[w2]MHz_wc_[wc]MHz/stage1/field_free

    Each runfolder must contain a file U.dat (resulting from propagation of
    pulse), and a file 'prop.log' that contain the output of the tm_en_prop
    program, including lines like

        eigen dist 00:   0.0000E+00 nrm:   0.0000E+00 En:  0.0000E+00  0.0000E+00 MHz
        eigen dist 01:   0.0000E+00 nrm:   3.7699E-05 En: -5.5301E-28 -6.0000E-03 MHz
        eigen dist 10:   6.8746E-05 nrm:   5.4840E-01 En:  8.7281E+01 -1.4310E-02 MHz
        eigen dist 11:   6.8808E-04 nrm:   1.0171E-01 En: -1.6188E+01 -2.4046E-02 MHz

    If T is given as the gate duration in ns, the gate stored in U.dat is
    compared to expected gate, and a warning is shown if they do not match.

    The resulting table will have the columns

    'w1 [GHz]'   : value of left qubit frequency
    'w2 [GHz]'   : value of right qubit frequency
    'wc [GHz]'   : value of cavity frequency
    'zeta [MHz]' : value of zeta
    'gamma [MHz]': Effective (population) decay rate for the qubit, due to
                   Purcell effect

    and use the runfolder path as an index
    """
    import logging
    logger = logging.getLogger(__name__)
    runfolders = []
    for folder in find_folders(runs, 'field_free'):
        U_dat = os.path.join(folder, 'U.dat')
        prop_log = os.path.join(folder, 'prop.log')
        if os.path.isfile(U_dat) and os.path.isfile(prop_log):
            runfolders.append(folder)
        else:
            logger.warn("folder %s is missing U.dat or prop.log" % folder)
    w1_s       = pd.Series(6.0, index=runfolders)
    w2_s       = pd.Series(index=runfolders)
    wc_s       = pd.Series(index=runfolders)
    zeta_s     = pd.Series(index=runfolders)
    gamma_s    = pd.Series(index=runfolders)
    errors = []
    two_pi = 2.0 * np.pi
    four_pi = 4.0 * np.pi
    for i, folder in enumerate(runfolders):
        w2, wc, E0, pulse_label = stage1_rf_to_params(folder)
        w2_s[i] = w2
        wc_s[i] = wc
        U_dat = os.path.join(folder, 'U.dat')
        prop_log = os.path.join(folder, 'prop.log')
        U = QDYN.gate2q.Gate2Q(U_dat)
        if np.isnan(U).any():
            logger.error("NaN in %s" % U_dat)
            errors.append(folder)
        elif np.max(U.logical_pops()) > 1.00001:
            logger.error("increase of norm in %s" % U_dat)
            errors.append(folder)
        with open(prop_log) as in_fh:
            E = {} # E['00'], E['01'], ... extracted from eigen_dist lines
            Gamma = {} # imaginary part of energy levels -> decay
            rx = re.compile(r'''
                eigen[ ]dist[ ](?P<state>[01]{2}):
                \s*[0-9.E+-]+ \s+ nrm: \s+ [0-9.E+-]+ \s+
                En: \s+ (?P<E>[0-9.E+-]+)
                \s+ (?P<MinusGamma>[0-9.E+-]+) \s+ MHz''', re.X)
            for line in in_fh:
                m = rx.match(line)
                if m:
                    state = m.group('state')
                    E[state] = float(m.group('E'))
                    Gamma[state] = -1.0*float(m.group('MinusGamma'))
        for state in ['00', '01', '10', '11']:
            assert state in E
        zeta_s[i] = E['00'] - E['01'] - E['10'] + E['11']
        gamma_s[i] = np.sum(np.array(Gamma.values(), dtype=np.float64)) / 2.0
        #          = 2 * (\sum_i \Gamma_i) / 4,
        #            with factor 2 due to amplitude -> population
        # check that gate is diagonal
        err = QDYN.linalg.norm(U-np.diag(np.diag(U)))
        if err > limit:
            logger.warn("Gate %s has non-diagonal, err = %g" % (U_dat, err))
        # check that phases are as expected
        if T is not None:
            observed_phases = np.angle(np.array([
                                U[0,0], U[1,1], U[2,2], U[3,3] ]))
            expected_phases = np.angle(np.array([
                 np.exp(-1j * two_pi * (E['00']/1000.0) * T),
                 np.exp(-1j * two_pi * (E['01']/1000.0) * T),
                 np.exp(-1j * two_pi * (E['10']/1000.0) * T),
                 np.exp(-1j * two_pi * (E['11']/1000.0) * T)]))
            err = QDYN.linalg.norm(observed_phases - expected_phases)
            if (err > limit*100):
                logger.warn("Phases in gate  %s have error of %g"
                            % (U_dat, err))
    table = pd.DataFrame(OrderedDict([
                ('w1 [GHz]',    w1_s),
                ('w2 [GHz]',    w2_s/1000.0),
                ('wc [GHz]',    wc_s/1000.0),
                ('zeta [MHz]',  zeta_s),
                ('gamma [MHz]', gamma_s),
            ]))
    return table[~table.index.isin(errors)]


def get_stage1_table(runs):
    """Summarize the results of the stage1 calculations in a DataFrame table.

    Assumes that the runfolder structure is

        [runs]/w2_[w2]MHz_wc_[wc]MHz/stage1/[pulse label]/E[E0 in MHz]

    except that the pulse amplitude subfolder is missing if the pulse label is
    'field free'.  Each runfolder must contain a file U.dat (resulting from
    propagation of pulse)

    The resulting table will have the columns

    'w1 [GHz]'  : value of left qubit frequency
    'w2 [GHz]'  : value of right qubit frequency
    'wc [GHz]'  : value of cavity frequency
    'C'         : Concurrence
    'avg loss', : Average loss from the logical subspace
    'max loss', : Maximal  loss from the logical subspace
    'E0 [MHz]'  : Peak amplitude of pulse
    'category'  : 'field_free', '1freq_center', '1freq_random',
                  '2freq_resonant', '2freq_random', '5freq_random'
    'J_PE'      : Value of functional for perfect-entangler target
    'J_SQ'      : Value of functional for single-qubit target
    'F_avg(unitary)': Value of average fidelity w.r.t the closest unitary

    and use the runfolder path as an index
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
    avg_loss_s = pd.Series(index=runfolders)
    max_loss_s = pd.Series(index=runfolders)
    E0_s       = pd.Series(index=runfolders)
    F_avg_s    = pd.Series(index=runfolders)
    category_s = pd.Series('', index=runfolders)
    errors = []
    for i, folder in enumerate(runfolders):
        w2, wc, E0, pulse_label = stage1_rf_to_params(folder)
        w2_s[i] = w2
        wc_s[i] = wc
        E0_s[i] = E0
        U_dat = os.path.join(folder, 'U.dat')
        U = QDYN.gate2q.Gate2Q(U_dat)
        if np.isnan(U).any():
            print "ERROR: NaN in %s" % U_dat
            errors.append(folder)
        elif np.max(U.logical_pops()) > 1.00001:
            print "ERROR: increase of norm in %s" % U_dat
            errors.append(folder)
        else:
            C_s[i] = U.closest_unitary().concurrence()
            avg_loss_s[i] = U.pop_loss()
            max_loss_s[i] = np.max(1.0 - U.logical_pops())
            F_avg_s[i] = U.F_avg(U.closest_unitary())
        category_s[i] = re.sub('_\d+$', '_random', pulse_label)
    table = pd.DataFrame(OrderedDict([
                ('w1 [GHz]', w1_s),
                ('w2 [GHz]', w2_s/1000.0),
                ('wc [GHz]', wc_s/1000.0),
                ('C',        C_s),
                ('avg loss', avg_loss_s),
                ('max loss', max_loss_s),
                ('E0 [MHz]', E0_s),
                ('category', category_s),
                ('J_PE',     J_PE(C_s, max_loss_s)),
                ('J_SQ',     J_SQ(C_s, max_loss_s)),
                ('F_avg(unitary)', F_avg_s),
            ]))
    return table[~table.index.isin(errors)]


def plot_zeta(zeta_table, T=50, scatter_size=0, outfile=None):
    """Plot zeta"""
    plots = PlotGrid()
    plots.scatter_size = scatter_size
    zeta = zeta_table['zeta [MHz]']
    w2 = zeta_table['w2 [GHz]']
    wc = zeta_table['wc [GHz]']
    plots.add_cell(w2, wc, np.abs(zeta), title='$\zeta [MHz]$', logscale=True)
    gamma = -2.0 * np.pi * (zeta/1000.0) * T
    C = np.abs(np.sin(0.5*gamma))
    plots.add_cell(w2, wc, C, vmin=0.0, vmax=1.0, title='concurrence')
    if outfile is None:
        plots.plot(quiet=True, show=True)
    else:
        fig = plots.plot(quiet=True, show=False)
        fig.savefig(outfile)
        plt.close(fig)


def plot_field_free_data(stage1_table, scatter_size=0, outfile=None):
    """Plot field-free concurrence"""
    plots = PlotGrid()
    plots.scatter_size = scatter_size
    table = stage1_table[stage1_table['category']=='field_free']
    plots.add_cell(table['w2 [GHz]'], table['wc [GHz]'], table['C'],
                   vmin=0.0, vmax=1.0, title='concurrence')
    plots.add_cell(table['w2 [GHz]'], table['wc [GHz]'], table['max loss'],
                   vmin=0.0, vmax=1.0, title='max population loss')
    if outfile is None:
        plots.plot(quiet=True, show=True)
    else:
        fig = plots.plot(quiet=True, show=False)
        fig.savefig(outfile)
        plt.close(fig)


def stage1_overview(T, rwa=True, inline=True, scatter_size=0, categories=None,
        table_loader=None, include_total=False):
    """Display a full report for the given gate duration"""
    frame = 'LAB'
    if rwa:
        frame = 'RWA'
    runfolder = './runs_{T:03d}_{frame}'.format(T=T, frame=frame)

    if table_loader is None:
        table_loader = get_stage1_table
    stage1_table = table_loader(runfolder)
    from select_for_stage2 import select_for_stage2
    t_PE = select_for_stage2(stage1_table, 'PE')
    t_SQ = select_for_stage2(stage1_table, 'SQ')

    display(Markdown('## T = %d ns (%s) ##' % (T, frame)))

    # Field-free
    if inline:
        display(Markdown('* Field-Free'))
        plot_field_free_data(stage1_table, scatter_size=scatter_size)
    else:
        outfile = 'stage1_field_free_{T:03d}_{frame}.png'.format(
                   T=T, frame=frame)
        plot_field_free_data(stage1_table, outfile=outfile)
        display(Markdown(dedent(r'''
        *   Field-Free

            Figure has been written to `{outfile}`
        '''.format(outfile=outfile))))

    # PE
    if inline:
        display(Markdown('* Selection for a Perfect Entangler'))
        plot_C_loss(t_PE, 'PE', loss_max=1.0, scatter_size=scatter_size,
                    categories=categories, include_total=include_total)
    else:
        outfile = 'stage1_PE_C_loss_{T:03d}_{frame}.png'.format(
                  T=T, frame=frame)
        plot_C_loss(t_PE, 'PE', loss_max=1.0, scatter_size=scatter_size,
                    categories=categories, outfile=outfile,
                    include_total=include_total)
        display(Markdown(dedent(r'''
        * Selection for a Perfect Entangler

            Figure has been written to `{outfile}`
        '''.format(outfile=outfile))))

    # SQ
    if inline:
        display(Markdown('* Selection for a Local Gate'))
        plot_C_loss(t_SQ, 'SQ', loss_max=1.0, scatter_size=scatter_size,
                    categories=categories, include_total=include_total)
    else:
        outfile = 'stage1_SQ_C_loss_{T:03d}_{frame}.png'.format(
                  T=T, frame=frame)
        plot_C_loss(t_SQ, 'SQ', loss_max=1.0, scatter_size=scatter_size,
                    categories=categories, outfile=outfile,
                    include_total=include_total)
        display(Markdown(dedent(r'''
        * Selection for a Local Gate

            Figure has been written to `{outfile}`
        '''.format(outfile=outfile))))


###############################################################################
#  Stage 2 Analysis
###############################################################################

def get_stage2_table(runs):
    """Summarize the results of the stage2 calculations in a DataFrame table

    Assumes that the runfolder structure is
    [runs]/w2_[w2]MHz_wc_[wc]MHz/stage2/[target]_[category]/

    Each runfolder must contain a file U.dat (resulting from
    propagation of pulse_opt.json), and a file U_closest_SQ.dat or
    U_closest_PE.dat, depending on the optimization target.

    The resulting table will have the columns

    'w1 [GHz]'  : value of left qubit frequency
    'w2 [GHz]'  : value of right qubit frequency
    'wc [GHz]'  : value of cavity frequency
    'C'         : Concurrence
    'avg loss', : Average loss from the logical subspace
    'max loss', : Maximum loss from the logical subspace
    'category'  : 'field_free', '1freq_center', '1freq_random',
                  '2freq_resonant', '2freq_random', '5freq_random'
    'target'    : 'PE', 'SQ'
    'J_PE'      : Value of functional for perfect-entangler target
    'J_SQ'      : Value of functional for single-qubit target
    'F_avg'     : Value of the average fidelity w.r.t the closest perfect
                  entangler or local gate (as appropriate)
    'c1'        : Weyl chamber coordinate c_1 for optimized gate (pi)
    'c2'        : Weyl chamber coordinate c_2 for optimized gate (pi)
    'c3'        : Weyl chamber coordinate c_3 for optimized gate (pi)
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
    F_avg_s    = pd.Series(index=runfolders)
    avg_loss_s = pd.Series(index=runfolders)
    max_loss_s = pd.Series(index=runfolders)
    category_s = pd.Series('', index=runfolders)
    target_s   = pd.Series('', index=runfolders)
    c1_s       = pd.Series(index=runfolders)
    c2_s       = pd.Series(index=runfolders)
    c3_s       = pd.Series(index=runfolders)
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
        U_closest_dat = os.path.join(folder, 'U_closest_SQ.dat')
        if not os.path.isfile(U_closest_dat):
            U_closest_dat = os.path.join(folder, 'U_closest_PE.dat')
        if os.path.isfile(U_closest_dat):
            U_closest = QDYN.gate2q.Gate2Q(U_closest_dat)
            F_avg_s[i] = U.F_avg(U_closest)
        c1, c2, c3 = U.closest_unitary().weyl_coordinates()
        c1_s[i] = c1
        c2_s[i] = c2
        c3_s[i] = c3
        C = U.closest_unitary().concurrence()
        C_s[i] = C
        avg_loss_s[i] = U.pop_loss()
        max_loss_s[i] = np.max(1.0 - U.logical_pops())
        category_s[i] = m_folder.group('category')
        target_s[i] = m_folder.group('target')
    table = pd.DataFrame(OrderedDict([
                ('w1 [GHz]', w1_s),
                ('w2 [GHz]', w2_s/1000.0),
                ('wc [GHz]', wc_s/1000.0),
                ('C',        C_s),
                ('avg loss', avg_loss_s),
                ('max loss', max_loss_s),
                ('category', category_s),
                ('target',   target_s),
                ('J_PE',     J_PE(C_s, max_loss_s)),
                ('J_SQ',     J_SQ(C_s, max_loss_s)),
                ('F_avg',    F_avg_s),
                ('c1',       c1_s),
                ('c2',       c2_s),
                ('c3',       c3_s),
            ]))
    return table


def get_Q_table(t_PE, t_SQ):
    """Combine two tables to calculate the overall "quality" function that
    expresses how well entanglement can both created and destroyed for a given
    parameter set. Both t_PE and t_SQ must have an F_avg column. The quality is
    simply the average of two correspoing F_avg values.

    Arguments
    ---------

    t_PE: pandas.DataFrame
        Table containing (at least) columns 'w1 [GHz]', 'w2 [GHz]', 'wc [GHz]',
        'category', 'F_avg'

    t_SQ: pandas.DataFrame
        Table containing (at least) columns 'w1 [GHz]', 'w2 [GHz]', 'wc [GHz]',
        'category', 'F_avg'

    Returns
    -------

    Q_table: pandas.DataFrame
        Table containing columns 'w1 [GHz]', 'w2 [GHz]', 'wc [GHz]',
        'category', 'F_avg (PE)', 'F_avg (SQ)', 'Q'
    """
    Q_table = pd.concat([
         t_PE.rename(columns={'F_avg': 'F_avg (PE)'})
         [['w1 [GHz]', 'w2 [GHz]', 'wc [GHz]', 'category', 'F_avg (PE)']]
         .set_index(['w1 [GHz]', 'w2 [GHz]', 'wc [GHz]', 'category']),
         t_SQ.rename(columns={'F_avg': 'F_avg (SQ)'})
         [['w1 [GHz]', 'w2 [GHz]', 'wc [GHz]', 'category', 'F_avg (SQ)']]
         .set_index(['w1 [GHz]', 'w2 [GHz]', 'wc [GHz]', 'category'])
        ], axis=1).reset_index()
    Q_table['Q'] = 0.5*(Q_table['F_avg (PE)'] + Q_table['F_avg (SQ)'])
    return Q_table


def show_weyl_chamber(table_PE, table_SQ, outfile=None):
    w_PE = WeylChamber()
    w_PE.scatter(table_PE['c1'], table_PE['c2'], table_PE['c3'])
    w_SQ = WeylChamber()
    w_SQ.scatter(table_SQ['c1'], table_SQ['c2'], table_SQ['c3'])
    fig = plt.figure(figsize=(13,5), dpi=72)
    ax_PE = fig.add_subplot(121, projection='3d', title="PE")
    ax_SQ = fig.add_subplot(122, projection='3d', title="SQ")
    w_PE.render(ax_PE)
    w_SQ.render(ax_SQ)
    if outfile is None:
        plt.show(fig)
    else:
        fig.savefig(outfile)
    plt.close(fig)


def write_quality_table(stage_table, outfile, category='1freq'):
    """Write out the achieved quality based on the given stage2 or stage3
    table, in tabular form (sorted)
    """
    if category is not None:
        stage_table = stage_table[stage_table['category']==category]
    (__, t_PE), (__, t_SQ) = stage_table.groupby('target', sort=True)
    Q_table = get_Q_table(t_PE, t_SQ).sort('Q', ascending=False)
    Q_table['Error'] = 1.0 - Q_table['Q']
    Q_table['Error'] = Q_table['Error'].map(lambda f: '%.3e'%f)
    with open(outfile, 'w') as out_fh:
        out_fh.write(Q_table.to_string())


def get_wd(configfile):
    """Return the driving frequency (in GHz) from the given config file"""
    rx = re.compile(r'w_d\s*=\s*(?P<w_d>[\d.]+)_MHz')
    with open(configfile) as in_fh:
        for line in in_fh:
            if line.strip().startswith('w_d'):
                w_d = float(rx.search(line).group('w_d'))
                return w_d / 1000.0 # to GHz
    raise AssertionError("Could not get w_d from config")

def get_prop_fig_table(stage4_table, stage4_folder='stage4',
        stage_prop_folder='stage_prop', what='pulse'):
    """Generate table of plots used in prop_overview routine"""
    index = stage4_table.index
    targets = ['PE_1freq', 'SQ_1freq_H_left', 'SQ_1freq_H_right',
                'SQ_1freq_Ph_left', 'SQ_1freq_Ph_right']
    err_col = {'PE_1freq'         : 'err(PE)',
               'SQ_1freq_H_left'  : 'err(H_L)',
               'SQ_1freq_H_right' : 'err(H_R)',
               'SQ_1freq_Ph_left' : 'err(S_L)',
               'SQ_1freq_Ph_right': 'err(S_R)'}
    fig_s = {}
    for target in targets:
        fig_s[target] = []
    for fld_stage4 in index: # e.g ./runs_050_RWA/w2_6200MHz_wc_5800MHz/stage4
        fld_prop = fld_stage4.replace(stage4_folder, stage_prop_folder)
        for target in targets:
            runfolder = os.path.join(fld_prop, target) # assumed to exist
            w_d = get_wd(os.path.join(runfolder, 'config'))
            err = stage4_table[err_col[target]][fld_stage4]
            title = "$\omega_d =$ %.3f GHz, $\epsilon = $%s" \
                    % (w_d, latex_float(err))
            if what == 'pulse':
                fig = new_figure(10.8, 8.0, quiet=True)
                fig.suptitle(title)
                p = QDYN.pulse.Pulse(os.path.join(runfolder, 'pulse.dat'))
                p.plot(fig=fig)
            elif what == 'popdyn':
                fig = new_figure(16.0, 12.0, quiet=True)
                dyn = QDYNTransmonLib.popdyn.PopPlot(
                          runfolder, panel_width=4.5, left_margin=2.5,
                          right_margin=3.0, top_margin=0.7)
                dyn.plot(basis_states=('00', '01', '10', '11'),
                         pops=('00', '01', '10', '11', 'tot'), fig=fig,
                         legend=True, in_panel_legend=True, quiet=True,
                         scale=1.0)
            else:
                raise ValueError("Unknown 'what'")
            fig_s[target].append(fig_to_html(fig))
            plt.close(fig)
    table = pd.DataFrame(OrderedDict([
                ('w1 [GHz]', stage4_table['w1 [GHz]']),
                ('w2 [GHz]', stage4_table['w2 [GHz]']),
                ('wc [GHz]', stage4_table['wc [GHz]']),
                ('pulse(PE)',      fig_s['PE_1freq']),
                ('pulse(H,left)',  fig_s['SQ_1freq_H_left']),
                ('pulse(H,right)', fig_s['SQ_1freq_H_right']),
                ('pulse(S,left)',  fig_s['SQ_1freq_Ph_left']),
                ('pulse(S,right)', fig_s['SQ_1freq_Ph_right']),
            ]))
    return table


def get_rho_prop_table(stage4_table, stage4_folder='stage4',
        stage_prop_folder='stage_prop_rho'):
    """Generate table comparing the gate error in Hilbert space to the gate
    error in Liouville space"""
    index = stage4_table.index
    targets = ['PE_1freq', 'SQ_1freq_H_left', 'SQ_1freq_H_right',
                'SQ_1freq_Ph_left', 'SQ_1freq_Ph_right']
    err_col = {'PE_1freq'         : 'err(PE)',
               'SQ_1freq_H_left'  : 'err(H_L)',
               'SQ_1freq_H_right' : 'err(H_R)',
               'SQ_1freq_Ph_left' : 'err(S_L)',
               'SQ_1freq_Ph_right': 'err(S_R)'}
    L_err = {}
    for target in targets:
        L_err[target] = pd.Series(index=index)
    for fld_stage4 in index:
        fld_prop = fld_stage4.replace(stage4_folder, stage_prop_folder)
        for target in targets:
            runfolder = os.path.join(fld_prop, target)
            err = None
            with open(os.path.join(runfolder, 'prop.log')) as in_fh:
                for line in in_fh:
                    m = re.search(r'F_avg\(L\)\s*:\s*([\d.Ee+-]+)', line)
                    if m:
                        err = 1.0 - float(m.group(1))
            L_err[target][fld_stage4] = err
    L_err['tot']  = pd.Series(index=index, data=0.0)
    for target in targets:
        L_err['tot'] += L_err[target]
    L_err['tot'] *= 1.0/len(targets)
    table = pd.DataFrame(OrderedDict([
                ('w1 [GHz]',    stage4_table['w1 [GHz]']),
                ('w2 [GHz]',    stage4_table['w2 [GHz]']),
                ('wc [GHz]',    stage4_table['wc [GHz]']),
                ('err_H(H_L)',  stage4_table['err(H_L)']),
                ('err_L(H_L)',  L_err['SQ_1freq_H_left']),
                ('err_H(S_L)',  stage4_table['err(S_L)']),
                ('err_L(S_L)',  L_err['SQ_1freq_Ph_left']),
                ('err_H(H_R)',  stage4_table['err(H_R)']),
                ('err_L(H_R)',  L_err['SQ_1freq_H_right']),
                ('err_H(S_R)',  stage4_table['err(S_R)']),
                ('err_L(S_R)',  L_err['SQ_1freq_Ph_right']),
                ('err_H(PE)',   stage4_table['err(PE)']),
                ('err_L(PE)',   L_err['PE_1freq']),
                ('err_H(tot)',  stage4_table['err(tot)']),
                ('err_L(tot)',  L_err['tot']),
                ('err_L/err_H(tot)',  L_err['tot']/stage4_table['err(tot)']),
            ]))
    return table


def prop_overview(T, rwa=True, err_limit=1.0e-3, stage4_folder='stage4',
        stage_prop_folder='stage_prop', rho=False, table_loader=None):
    """Find all stage_prop folders for the given pulse duration and plot their
    results. Pulses and population dynamics are only shown for total errors
    below the given limit"""
    frame = 'LAB'
    if rwa:
        frame = 'RWA'
    root = './runs_{T:03d}_{frame}'.format(T=T, frame=frame)
    display(Markdown('## T = %d ns (%s) ##' % (T, frame)))
    if table_loader is None:
        table_loader = get_stage4_table
    stage_table = table_loader(root, stage_folder=stage4_folder)

    formatters = { # col value -> HTML
        'err(H_L)': latex_float,
        'err(H_R)': latex_float,
        'err(S_L)': latex_float,
        'err(S_R)': latex_float,
        'err(PE)' : latex_float,
        'err(tot)': latex_float,
        'err_H(H_L)': latex_float,
        'err_L(H_L)': latex_float,
        'err_H(S_L)': latex_float,
        'err_L(S_L)': latex_float,
        'err_H(H_R)': latex_float,
        'err_L(H_R)': latex_float,
        'err_H(S_R)': latex_float,
        'err_L(S_R)': latex_float,
        'err_H(PE)':  latex_float,
        'err_L(PE)':  latex_float,
        'err_H(tot)': latex_float,
        'err_L(tot)': latex_float,
    }

    with pd.option_context('display.max_colwidth', -1):

        if rho:
            display(Markdown("\nComparison of Gate Errors in Hilbert "
            "space and Liouville space:\n"))
            err_table = get_rho_prop_table(stage_table,
                            stage4_folder=stage4_folder,
                            stage_prop_folder=stage_prop_folder
                        ).sort('err_H(tot)')
            html = err_table.to_html(formatters=formatters,
                                    escape=False, index=False)
            html = pd_insert_col_width(html, widths=([50,]*3+[100,]*12))
            display(HTML(html))
        else:
            display(Markdown("\nAchieved Gate Error for all optimizations, with "
            "total error 'err(tot)' as average of the error for all targets:\n"))
            err_table = stage_table.sort('err(tot)')
            html = err_table.to_html(formatters=formatters,
                                    escape=False, index=False)
            html = pd_insert_col_width(html, widths=([50,]*3+[100,]*7))
            display(HTML(html))

            display(Markdown("\nOptimized pulses for optimizations with err(tot)"
                            " < %s:\n" % latex_float(err_limit)))
            pulse_fig_table = get_prop_fig_table(
                            stage_table[stage_table['err(tot)']<err_limit]
                            .sort('err(tot)'), stage4_folder=stage4_folder,
                            stage_prop_folder=stage_prop_folder, what='pulse')
            html = pulse_fig_table.to_html(formatters=formatters,
                                        escape=False, index=False)
            html = pd_insert_col_width(html, widths=([50,]*3+[300,]*5))
            display(HTML(html))

            display(Markdown("Population dynamics for optimizations with err(tot)"
                            " < %s:\n" % latex_float(err_limit)))
            popdyn_fig_table = get_prop_fig_table(
                            stage_table[stage_table['err(tot)']<err_limit]
                            .sort('err(tot)'), stage4_folder=stage4_folder,
                            stage_prop_folder=stage_prop_folder, what='popdyn')
            html = popdyn_fig_table.to_html(formatters=formatters,
                                            escape=False, index=False)
            html = pd_insert_col_width(html, widths=([50,]*3+[400,]*5))
            display(HTML(html))


def oct_overview(T, stage, rwa=True, inline=True, scatter_size=0,
        categories=None, table_loader=None, include_total=False):
    """Display a full report for the given gate duration"""
    assert stage in ['stage2', 'stage3'], \
    "stage must be in ['stage1', 'stage2']"
    frame = 'LAB'
    if rwa:
        frame = 'RWA'
    runfolder = './runs_{T:03d}_{frame}'.format(T=T, frame=frame)

    if table_loader is None:
        if stage == 'stage2':
            table_loader = get_stage2_table
        else:
            table_loader = get_stage3_table
    table = table_loader(runfolder)
    (__, t_PE), (__, t_SQ) = table.groupby('target', sort=True)

    display(Markdown('## T = %d ns (%s) ##' % (T, frame)))

    # Concurrence and loss -- PE
    if inline:
        display(Markdown('* Concurrence and Loss -- PE'))
        plot_C_loss(t_PE, 'PE', categories=categories,
                    scatter_size=scatter_size, include_total=include_total)
    else:
        outfile = '{stage}_PE_C_loss_{T:03d}_{frame}.png'.format(
                   stage=stage, T=T, frame=frame)
        plot_C_loss(t_PE, 'PE', categories=categories,
                    scatter_size=scatter_size, include_total=include_total,
                    outfile=outfile)
        display(Markdown(dedent(r'''
        * Concurrence and Loss -- PE

            Figure has been written to `{outfile}`
        '''.format(outfile=outfile))))

    # Quality
    min_err = diss_error(gamma=1.2e-5, t=T)
    if inline:
        display(Markdown('* Quality'))
        plot_quality(t_PE, t_SQ, include_total=include_total, vmin=min_err,
                     scatter_size=scatter_size, categories=categories)
    else:
        outfile = '{stage}_quality_{T:03d}_{frame}.png'.format(
                   stage=stage, T=T, frame=frame)
        plot_quality(t_PE, t_SQ, include_total=include_total, vmin=min_err,
                     scatter_size=scatter_size, categories=categories,
                     outfile=outfile)
        display(Markdown(dedent(r'''
        * Quality

            Figure has been written to `{outfile}`
        '''.format(outfile=outfile))))

    # Weyl chamber
    t_PE_weyl = t_PE[(t_PE['max loss']<0.1) & (t_PE['C']==1.0)]
    t_SQ_weyl = t_SQ[t_SQ['max loss']<0.1]
    if inline:
        display(Markdown('* Weyl Chamber'))
        show_weyl_chamber(t_PE_weyl, t_SQ_weyl)
    else:
        outfile = '{stage}_weyl_{T:03d}_{frame}.png'.format(
                   stage=stage, T=T, frame=frame)
        show_weyl_chamber(t_PE_weyl, t_SQ_weyl, outfile=outfile)
        display(Markdown(dedent(r'''
        * Weyl chamber

            Figure has been written to `{outfile}`
        '''.format(outfile=outfile))))


def show_oct_summary_table(gate_times=None, rwa=True, stage1_table_reader=None,
        oct_table_reader=None):
    """Show a table with the following columns:

    min f-free error: error with which the field-free evolution implements the
        closest unitary gate (over all points in the parameter space)
    f-free error: error with which the field-free evolution implements the
        closest unitary gate, at the parameter point (w_2, w_c) at which the
        quality is maximal (see w2, wc column)
    PE error: smallest error with which an arbitrary perfect entangler can be
        implemented (over all points in the parameter space)
    Q error: smallest quality error (avg of PE and SQ error) than can be
        achieved
    w2 [GHz]: Value of w_2 at which the Q error is minimal
    wc [GHz]: Value of w_c at which the Q error is minimal
    """
    frame = 'LAB'
    if rwa:
        frame = 'RWA'
    if gate_times is None:
        gate_times = [5, 10, 20, 50, 100, 200] # ns
    predicted_min_error = {}
    min_field_free_error = {}
    field_free_error = {}
    achieved_PE_error = {}
    achieved_Q_error = {}
    w2_val = {}
    wc_val = {}
    if oct_table_reader is None:
        oct_table_reader = get_stage2_table
    if stage1_table_reader is None:
        stage1_table_reader = get_stage1_table
    for T in gate_times:
        runfolder = './runs_{T:03d}_{frame}'.format(T=T, frame=frame)
        stage1_table = stage1_table_reader(runfolder)
        stage1_table = stage1_table[stage1_table['category']=='field_free']
        oct_table = oct_table_reader(runfolder)
        (__, t_PE), (__, t_SQ) = oct_table.groupby('target', sort=True)
        Q_table = get_Q_table(t_PE, t_SQ)
        Q_table = Q_table[Q_table['category'].str.contains('1freq')]
        predicted_min_error[T] = diss_error(gamma=1.2e-5, t=T)
        min_field_free_error[T] = 1.0 - stage1_table['F_avg(unitary)'].max()
        i = Q_table['Q'].idxmax()
        w2_val[T] = Q_table['w2 [GHz]'][i]
        wc_val[T] = Q_table['wc [GHz]'][i]
        achieved_Q_error[T] = 1.0 - Q_table['Q'][i]
        achieved_PE_error[T] = 1.0 - Q_table['F_avg (PE)'].max()
        field_free_error[T] = 1.0 - stage1_table[
                                      (stage1_table['wc [GHz]'] == wc_val[T])\
                                    & (stage1_table['w2 [GHz]'] == w2_val[T])
                              ]['F_avg(unitary)'].max()

    df = pd.DataFrame(index=gate_times,
        data=OrderedDict([
            #('predicted min error', [diss_error(gamma=1.2e-5, t=t) for t in gate_times]),
            ('min f-free err', [min_field_free_error[t] for t in gate_times]),
            ('f-free err', [field_free_error[t] for t in gate_times]),
            ('PE error', [achieved_PE_error[t] for t in gate_times]),
            ('Q error', [achieved_Q_error[t] for t in gate_times]),
            ('w2 [GHz]', [w2_val[t] for t in gate_times]),
            ('wc [GHz]', [wc_val[t] for t in gate_times]),
            ]))
    df.index.name = "gate duration [ns]"
    print df.to_string(float_format=lambda f:'%.2e'%f)
    #print df.to_latex(float_format=lambda r: latex_float(r), escape=False)


###############################################################################
#  Stage 3 Analysis
###############################################################################

def get_stage3_input_table(runs):
    from select_for_stage3 import select_for_stage3
    stage2_table = get_stage2_table(runs)
    stage3_input_table = {}
    for target in ['PE', 'SQ']:
        stage3_input_table[target] \
        = select_for_stage3(stage2_table, target=target)
        stage3_input_table[target]['stage2 runfolder'] \
        = stage3_input_table[target].index
        stage3_runfolders = pd.Series(index=stage3_input_table[target].index)
        for stage2_runfolder, row in stage3_input_table[target].iterrows():
            stage3_runfolder = os.path.join(runs,
                                'w2_%dMHz_wc_%dMHz' % (row['w2 [GHz]']*1000,
                                                    row['wc [GHz]']*1000),
                                'stage3',
                                '%s_%s' % (target, row['category']))
            stage3_runfolders[stage2_runfolder] = stage3_runfolder
        stage3_input_table[target].set_index(stage3_runfolders, inplace=True)
    return stage3_input_table['PE'].append(stage3_input_table['SQ'])


def get_stage3_table(runs):
    """Summarize the results of the stage3 calculations in a DataFrame table

    Assumes that the runfolder structure is
    [runs]/w2_[w2]MHz_wc_[wc]MHz/stage3/[target]_[category]/

    Each runfolder must contain a file U.dat (resulting from
    propagation of pulse_opt.json), and a file U_closest_SQ.dat or
    U_closest_PE.dat, depending on the optimization target.

    The resulting table will have the columns

    'w1 [GHz]'         : value of left qubit frequency
    'w2 [GHz]'         : value of right qubit frequency
    'wc [GHz]'         : value of cavity frequency
    'C'                : Concurrence
    'avg loss',        : average loss from the logical subspace
    'max loss',        : max loss from the logical subspace
    'category'         : '1freq', '2freq', '5freq'
    'target'           : 'PE', 'SQ'
    'F_avg'            : Value of the average fidelity w.r.t the closest perfect
                         entangler or local gate (as appropriate)
    'J_PE'             : Value of functional for perfect-entangler target
    'J_SQ'             : Value of functional for single-qubit target
    'c1'               : Weyl chamber coordinate c_1 for optimized gate (pi)
    'c2'               : Weyl chamber coordinate c_2 for optimized gate (pi)
    'c3'               : Weyl chamber coordinate c_3 for optimized gate (pi)
    'stage2 runfolder' : Runfolder from which guess pulse originates
    'C (guess)'        : Concurrence of guess pulse
    'avg loss (guess)' : Avergage loss for guess pulse
    'max loss (guess)' : Max loss for guess pulse
    'J_PE (guess)'     : Value of J_PE for guess pulse
    'J_SQ (guess)'     : Value of J_SQ for guess pulse
    'F_avg (guess)'    : Value of F_avg for guess pulse

    The index is given by the full runfolder path
    """
    runfolders = []
    for folder in find_folders(runs, 'stage3'):
        for subfolder in find_leaf_folders(folder):
            if os.path.isfile(os.path.join(subfolder, 'U.dat')):
                runfolders.append(subfolder)
    w1_s       = pd.Series(6.0, index=runfolders)
    w2_s       = pd.Series(index=runfolders)
    wc_s       = pd.Series(index=runfolders)
    C_s        = pd.Series(index=runfolders)
    F_avg_s    = pd.Series(index=runfolders)
    avg_loss_s = pd.Series(index=runfolders)
    max_loss_s = pd.Series(index=runfolders)
    category_s = pd.Series('', index=runfolders)
    target_s   = pd.Series('', index=runfolders)
    c1_s       = pd.Series(index=runfolders)
    c2_s       = pd.Series(index=runfolders)
    c3_s       = pd.Series(index=runfolders)
    rx_folder = re.compile(r'''
                \/w2_(?P<w2>[\d.]+)MHz_wc_(?P<wc>[\d.]+)MHz
                \/stage3
                \/(?P<target>PE|SQ)
                  _(?P<category>1freq|2freq|5freq)
                ''', re.X)
    for i, folder in enumerate(runfolders):
        m_folder = rx_folder.search(folder)
        if not m_folder:
            raise ValueError("%s does not match rx_folder" % folder)
        w2_s[i] = float(m_folder.group('w2'))
        wc_s[i] = float(m_folder.group('wc'))
        U_dat = os.path.join(folder, 'U.dat')
        U = QDYN.gate2q.Gate2Q(U_dat)
        U_closest_dat = os.path.join(folder, 'U_closest_SQ.dat')
        if not os.path.isfile(U_closest_dat):
            U_closest_dat = os.path.join(folder, 'U_closest_PE.dat')
        if os.path.isfile(U_closest_dat):
            U_closest = QDYN.gate2q.Gate2Q(U_closest_dat)
            F_avg_s[i] = U.F_avg(U_closest)
        c1, c2, c3 = U.closest_unitary().weyl_coordinates()
        c1_s[i] = c1
        c2_s[i] = c2
        c3_s[i] = c3
        C = U.closest_unitary().concurrence()
        C_s[i] = C
        avg_loss_s[i] = U.pop_loss()
        max_loss_s[i] = np.max(1.0 - U.logical_pops())
        category_s[i] = m_folder.group('category')
        target_s[i] = m_folder.group('target')
    table = pd.DataFrame(OrderedDict([
                ('w1 [GHz]', w1_s),
                ('w2 [GHz]', w2_s/1000.0),
                ('wc [GHz]', wc_s/1000.0),
                ('C',        C_s),
                ('avg loss', avg_loss_s),
                ('max loss', max_loss_s),
                ('category', category_s),
                ('target',   target_s),
                ('J_PE',     J_PE(C_s, max_loss_s)),
                ('J_SQ',     J_SQ(C_s, max_loss_s)),
                ('F_avg',    F_avg_s),
                ('c1',       c1_s),
                ('c2',       c2_s),
                ('c3',       c3_s),
            ]))
    input_table = get_stage3_input_table(runs)
    table['stage2 runfolder'] = input_table['stage2 runfolder']
    table['C (guess)']        = input_table['C']
    table['avg loss (guess)'] = input_table['avg loss']
    table['max loss (guess)'] = input_table['max loss']
    table['J_PE (guess)']     = input_table['J_PE']
    table['J_SQ (guess)']     = input_table['J_SQ']
    table['F_avg (guess)']    = input_table['F_avg']
    return table


def read_target_gate(filename):
    """Return a QDYN.gate2q.Gate2Q instance for the the gate defined in the
    given filename, assuming the file contains a gate in the format that
    select_for_stage4.preparae_stage4 uses for the file  'target_gate.dat'.
    Specifically, the file is assumed to contain two columns for the real and
    the imaginary part of the vectorized gate. The vectorization is assumed to
    be in column-major mode (columns of the matrix written underneath each
    other)
    """
    gate_re, gate_im = np.genfromtxt(filename, unpack=True)
    gate = np.reshape((gate_re + 1j*gate_im), newshape=(4,4), order='F')
    return QDYN.gate2q.Gate2Q(gate)


def get_stage4_table(runs, stage_folder):
    """Summarize the results of the stage4 calculations in a DataFrame table.
    The default value for `stage_folder` should be 'stage4'.

    Assumes that the runfolder structure is
    [runs]/w2_[w2]MHz_wc_[wc]MHz/[stage_folder]/[SQ|PE]_[category]_[gate]/

    Each runfolder must contain a file U.dat (resulting from
    ropagation of the optimized pulse.dat), and a file target_gate.dat that
    contains the optimization target.

    The resulting table will have the columns

    'w1 [GHz]' : value of left qubit frequency
    'w2 [GHz]' : value of right qubit frequency
    'wc [GHz]' : value of cavity frequency
    'err(H_L)' : avg gate error for Hadamard gate on left qubit
    'err(S_L)' : avg gate error for Phase gate on left qubit
    'err(H_R)' : avg gate error for Hadamard gate on right qubit
    'err(S_R)' : avg gate error for Phase gate on right qubit
    'err(PE)':   avg gate error for PE gate
    'err(tot)' : average of all errors

    The index is given by the full stage4 path corresponding to a tuple
    (w1, w2, wc)
    """
    # `stage_folder` has no default value, to that memoization isn't too
    # confusing
    stage4_folders = list(find_folders(runs, stage_folder))
    w1_s       = pd.Series(6.0, index=stage4_folders)
    w2_s       = pd.Series(index=stage4_folders)
    wc_s       = pd.Series(index=stage4_folders)
    err_H_L_s  = pd.Series(index=stage4_folders)
    err_S_L_s  = pd.Series(index=stage4_folders)
    err_H_R_s  = pd.Series(index=stage4_folders)
    err_S_R_s  = pd.Series(index=stage4_folders)
    err_PE_s   = pd.Series(index=stage4_folders)
    err_tot    = pd.Series(index=stage4_folders)
    rx_stage4_folder = re.compile(r'''
                \/w2_(?P<w2>[\d.]+)MHz_wc_(?P<wc>[\d.]+)MHz
                \/'''+stage_folder, re.X)
    err = { # map between part of runfolder name and the data series
        'H_left':   err_H_L_s,
        'H_right':  err_H_R_s,
        'Ph_left':  err_S_L_s,
        'Ph_right': err_S_R_s,
        'PE':       err_PE_s
    }
    for i, stage4_folder in enumerate(stage4_folders):
        m_stage4_folder = rx_stage4_folder.search(stage4_folder)
        if not m_stage4_folder:
            raise ValueError("%s does not match rx_stage4_folder" % folder)
        w2_s[i] = float(m_stage4_folder.group('w2'))
        wc_s[i] = float(m_stage4_folder.group('wc'))
        for runfolder in find_leaf_folders(stage4_folder):
            assert '1freq' in runfolder
            processed = False
            for target in err:
                if target in runfolder:
                    processed = True
                    try:
                        U_dat = os.path.join(runfolder, 'U.dat')
                        U = QDYN.gate2q.Gate2Q(file=U_dat)
                        target_gate_dat = os.path.join(runfolder,
                                                        'target_gate.dat')
                        U_target = read_target_gate(target_gate_dat)
                        err[target][i] = 1.0 - U.F_avg(U_target)
                    except IOError:
                        pass # U.dat doesn't exist => Leave NaN in output table
            assert processed
    err_tot = (err_H_L_s + err_H_R_s + err_S_L_s + err_S_R_s + err_PE_s) / 5.0
    table = pd.DataFrame(OrderedDict([
                ('w1 [GHz]',  w1_s),
                ('w2 [GHz]',  w2_s/1000.0),
                ('wc [GHz]',  wc_s/1000.0),
                ('err(H_L)',  err_H_L_s),
                ('err(S_L)',  err_H_R_s),
                ('err(H_R)',  err_S_L_s),
                ('err(S_R)',  err_S_R_s),
                ('err(PE)',   err_PE_s),
                ('err(tot)',  err_tot)
            ]))
    return table


def add_1freq_wL_E0(table, runfolder_col=None, pulse_file='pulse_opt.json'):
    """Return a copy of table, filtered to category starting with '1freq', with
    two added columns ('E0', 'w_L')"""
    E0s = pd.Series(index=table.index)
    wLs = pd.Series(index=table.index) - table['w1 [GHz]']
    table = table[table['category']\
            .isin(['1freq', '1freq_center', '1freq_random'])].copy()
    runfolders = table.index
    if runfolder_col is not None:
        runfolders = table[runfolder_col]
    else:
        runfolders = pd.Series(table.index, index=table.index)
    for i, row in table.iterrows():
        runfolder = runfolders[i]
        p = AnalyticalPulse.read(os.path.join(runfolder, pulse_file))
        E0s[i] = p.parameters['E0']
        wLs[i] = p.parameters['w_L']
    table['E0'] = E0s
    table['w_L'] = wLs
    return table


def plot_wL_E0(stage3_table, outfile=None):
    table = add_1freq_wL_E0(stage3_table, runfolder_col='stage2 runfolder',
                            pulse_file='pulse_opt.json')
    plots = PlotGrid()
    w_center = 0.5*(table['w1 [GHz]'] + table['w2 [GHz]'])
    contours = []
    plots.add_cell(table['w2 [GHz]'], table['wc [GHz]'],
            table['w_L'] - table['w1 [GHz]'],
            val_alpha=(1-table['max loss']), bg='white',
            vmin=-1.0, vmax=1.0, contour_levels=contours,
            logscale=False, title="w_L - w1 (GHz)")
    plots.add_cell(table['w2 [GHz]'], table['wc [GHz]'],
            table['w_L'] - table['w2 [GHz]'],
            val_alpha=(1-table['max loss']), bg='white',
            vmin=-1.0, vmax=1.0, contour_levels=contours,
            logscale=False, title="w_L - w2 (GHz)")
    plots.add_cell(table['w2 [GHz]'], table['wc [GHz]'],
            table['w_L'] - w_center,
            val_alpha=(1-table['max loss']), bg='white',
            vmin=-1.0, vmax=1.0, contour_levels=contours,
            logscale=False, title="w_L - w_center (GHz)")
    plots.add_cell(table['w2 [GHz]'], table['wc [GHz]'],
            table['w_L'] - table['wc [GHz]'],
            val_alpha=(1-table['max loss']), bg='white',
            vmin=-1.0, vmax=1.0, contour_levels=contours,
            logscale=False, title="w_L - wc (GHz)")
    plots.add_cell(table['w2 [GHz]'], table['wc [GHz]'],
            table['E0'], val_alpha=(1-table['max loss']), bg='white',
            vmin=0, vmax=1500.0, contour_levels=16,
            logscale=False, title="E0 (MHz)")
    if outfile is None:
        plots.plot(quiet=True, show=True)
    else:
        fig = plots.plot(quiet=True, show=False)
        fig.savefig(outfile)
        plt.close(fig)


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
    elif analytical_pulse.formula_name in ['1freq', '1freq_rwa']:
        return p['w_L']
    elif analytical_pulse.formula_name in ['2freq', '2freq_rwa']:
        s = abs(p['a_1']) + abs(p['a_2'])
        return (p['freq_1'] * abs(p['a_1']) + p['freq_2'] * abs(p['a_2']))/s
    elif analytical_pulse.formula_name in ['5freq', '5freq_rwa']:
        weights = np.sqrt(np.abs(p['a_high'])**2 + np.abs(p['b_high'])**2)
        weights *= 1.0/np.sum(weights)
        return np.sum(weights * p['freq_high'])
    elif analytical_pulse.formula_name in ['CRAB_rwa',]:
        return p['w_d']
    else:
        raise ValueError("Unknown formula name")


def max_freq_delta(analytical_pulse, w_L):
    """Return the maximum frequency that must be resolved for the given pulse
    in a rotating frame wL. All frequencies in GHz."""
    import numpy as np
    p = analytical_pulse.parameters
    if analytical_pulse.formula_name == 'field_free':
        return 0.0
    elif analytical_pulse.formula_name in ['1freq', '1freq_rwa']:
        return abs(w_L - p['w_L'])
    elif analytical_pulse.formula_name in ['2freq', '2freq_rwa']:
        return max(abs(w_L - p['freq_1']),  abs(w_L - p['freq_2']))
    elif analytical_pulse.formula_name in ['5freq', '5freq_rwa']:
        return np.max(np.abs(w_L - p['freq_high']))
    elif analytical_pulse.formula_name in ['CRAB_rwa',]:
        assert (w_L == p['w_d'])
        return 0.0
    else:
        raise ValueError("Unknown formula name")



def prop_RWA(config_file, pulse_json, outfolder, runfolder=None):
    """Given a config file and pulse file in the lab frame, (temporarily)
    modify them to be in the RWA and propagate. The propagation will take place
    in the given runfolder. If no runfolder is given, create a temporary
    runfolder which will be deleted after the propagation has finished. The
    file U.dat resulting from the propagation is copied to the given outfolder,
    as U_RWA.dat. Also, a file 'rwa_info.dat' is also written to the outfolder,
    detailing some of the parameters of the rotating frame. The outfolder may
    be identical to the runfolder. However, runfolder should not be the folder
    in which config_file and pulse_json are located, as these files would be
    overwritten with the RWA version otherwise.
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
            cmds = []
            cmds.append(['tm_en_gh', '--rwa', '--dissipation', '.'])
            cmds.append(['rewrite_dissipation.py',])
            cmds.append(['tm_en_logical_eigenstates.py', '.'])
            cmds.append(['tm_en_prop', '.'])
            for cmd in cmds:
                stdout.write("**** " + " ".join(cmd) +"\n")
                sp.call(cmd, cwd=temp_runfolder, env=env,
                        stderr=sp.STDOUT, stdout=stdout)
            end = time.time()
            stdout.write("**** finished in %s seconds . \n"%(end-start))
            rwa_info += "propagation time: %d seconds\n" % (end-start)
            copy(join(temp_runfolder, 'U.dat'), join(outfolder, 'U_RWA.dat'))
            copy(join(temp_runfolder, 'prop.log'),
                 join(outfolder, 'prop_rwa.log'))
        write_file(join(outfolder, 'rwa_info.dat'), rwa_info)
    except Exception as e:
        print e
    finally:
        if runfolder is None:
            rmtree(temp_runfolder)


def prop_LAB(config_file, pulse_json, outfolder, runfolder=None):
    """Given a config file and pulse file in the rotating frame, (temporarily)
    modify them to be in the lab frame and propagate. The propagation will take
    place in the given runfolder. If no runfolder is given, create a temporary
    runfolder which will be deleted after the propagation has finished. The
    file U.dat resulting from the propagation is copied to the given outfolder,
    as U_LAB.dat. The outfolder may be identical to the runfolder. However,
    runfolder should not be the folder in which config_file and pulse_json are
    located, as these files would be overwritten with the lab frame version
    otherwise.
    """
    import os
    import re
    from os.path import join
    from analytical_pulses import AnalyticalPulse
    from clusterjob.utils import read_file, write_file
    from QDYN.shutil import mkdir, copy, rmtree
    import time
    import subprocess as sp
    import uuid
    import logging

    p = AnalyticalPulse.read(pulse_json)
    config = read_file(config_file)

    logger = logging.getLogger(__name__)

    if not p.formula_name.endswith('_rwa'):
        raise ValueError("Formula name in %s must end with _rwa" % pulse_json)
    nt = 1100 * int(p.T)
    p.nt = nt
    p._formula = p.formula_name.replace('_rwa', '')
    del p.parameters['w_d']

    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '4'

    if runfolder is None:
        temp_runfolder = join(os.environ['SCRATCH_ROOT'], str(uuid.uuid4()))
        logger.debug("Using temp runfolder %s", temp_runfolder)
    else:
        temp_runfolder = runfolder
    mkdir(temp_runfolder)
    mkdir(outfolder)
    try:
        config = re.sub('w_d\s*=\s*[\d.]+_MHz', 'w_d = 0.0_MHz',
                        config)
        config = re.sub('nt\s*=\s*\d+', 'nt = %d'%nt,
                        config)
        config = re.sub('prop_guess\s*=\s*F', 'prop_guess = T', config)

        config = re.sub('is_complex\s*=\s*T', 'is_complex = F', config)

        write_file(join(temp_runfolder, 'config'), config)
        p.write(join(temp_runfolder, 'pulse.guess.json'), pretty=True)
        pulse = p.pulse(time_unit='ns', ampl_unit='MHz')
        pulse.write(join(temp_runfolder, 'pulse.guess'))
        start = time.time()
        with open(join(temp_runfolder, 'prop.log'), 'w', 0) as stdout:
            cmds = []
            cmds.append(['tm_en_gh', '--dissipation', '.'])
            cmds.append(['rewrite_dissipation.py',])
            cmds.append(['tm_en_logical_eigenstates.py', '.'])
            cmds.append(['tm_en_prop', '.'])
            for cmd in cmds:
                stdout.write("**** " + " ".join(cmd) +"\n")
                sp.call(cmd, cwd=temp_runfolder,
                        stderr=sp.STDOUT, stdout=stdout)
            end = time.time()
            stdout.write("**** finished in %s seconds . \n"%(end-start))
            copy(join(temp_runfolder, 'U.dat'), join(outfolder, 'U_LAB.dat'))
            copy(join(temp_runfolder, 'prop.log'),
                 join(outfolder, 'prop_lab.log'))
    except Exception as e:
        print e
    finally:
        if runfolder is None:
            logger.debug("Removing temp runfolder %s", temp_runfolder)
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
                cmds = []
                cmds.append(['tm_en_gh', '--dissipation', '.'])
                cmds.append(['rewrite_dissipation.py',])
                cmds.append(['tm_en_logical_eigenstates.py', '.'])
                cmds.append(['tm_en_prop', '.'])
                for cmd in cmds:
                    stdout.write("**** " + " ".join(cmd) +"\n")
                    sp.call(cmd, cwd=lab_runfolder,
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


def get_LAB_table(runs):
    """Summarize the results of the stage2 LAB propagation

    Looks for U_RWA.dat in all the subfolders of the given `runs` folder.
    A file U.dat obtained from propagating in the rotating frame must also be
    present in the same folder

    The resulting table will have the columns

    'C (RWA)'              : Concurrence in RWA propagation
    'loss (RWA)',          : Loss from the log. subspace in RWA propagation
    'C (LAB)'              : Concurrence in LAB propagation
    'loss (LAB)',          : Loss from the log. subspace in LAB propagation
    'Delta C'              : Absolute value of difference (concurrence)
    'Delta loss'           : Absolute value of difference (loss)

    and use the runfolder name as the index
    """
    runfolders = []
    for U_RWA_dat in find_files(runs, 'U_LAB.dat'):
        U_LAB_dat = U_RWA_dat.replace("_LAB", "")
        if os.path.isfile(U_LAB_dat):
            runfolders.append(os.path.split(U_RWA_dat)[0])
    C_RWA_s         = pd.Series(index=runfolders)
    loss_RWA_s      = pd.Series(index=runfolders)
    C_LAB_s         = pd.Series(index=runfolders)
    loss_LAB_s      = pd.Series(index=runfolders)
    for i, folder in enumerate(runfolders):
        U_RWA_dat = os.path.join(folder, 'U.dat')
        U_LAB_dat = os.path.join(folder, 'U_LAB.dat')
        U_RWA = QDYN.gate2q.Gate2Q(U_RWA_dat)
        U_LAB = QDYN.gate2q.Gate2Q(U_LAB_dat)
        C_RWA_s[i]    = U_RWA.closest_unitary().concurrence()
        loss_RWA_s[i] = U_RWA.pop_loss()
        C_LAB_s[i]    = U_LAB.closest_unitary().concurrence()
        loss_LAB_s[i] = U_LAB.pop_loss()
    table = pd.DataFrame(OrderedDict([
                ('C (RWA)',               C_RWA_s),
                ('loss (RWA)',            loss_RWA_s),
                ('C (LAB)',               C_LAB_s),
                ('loss (LAB)',            loss_LAB_s),
                ('Delta C',               (C_RWA_s-C_LAB_s).abs()),
                ('Delta loss',            (loss_RWA_s-loss_LAB_s).abs()),
            ]))
    return table


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
