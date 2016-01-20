#!/usr/bin/env python
import os
import sys
from StringIO import StringIO
import QDYN
import QDYNTransmonLib
import numpy as np
import matplotlib
from collections import OrderedDict
matplotlib.use('Agg')
import matplotlib.pylab as plt
from notebook_utils import get_Q_table, diss_error, PlotGrid
from notebook_utils import (get_stage1_table, get_stage2_table,
        get_stage3_table, get_zeta_table)
from mgplottools.mpl import new_figure, set_axis
from matplotlib.ticker import FuncFormatter

STYLE = 'paper.mplstyle'

get_stage3_table = QDYN.memoize.memoize(get_stage3_table)
get_stage3_table.load('stage3_table.cache')
get_zeta_table = QDYN.memoize.memoize(get_zeta_table)
get_zeta_table.load('zeta_table.cache')

OUTFOLDER = './paper_images'
OUTFOLDER = '/Users/goerz/Documents/Papers/TransmonLandscape'


def generate_field_free_plot(zeta_table, T, outfile):
    """Plot field-free entangling energy zeta, and projected concurrence after
    the given gate duration T in ns.
    """
    plots = PlotGrid(layout='paper')
    # parameters matching those in generate_map_plot
    plots.cell_width      =  6.375
    plots.cell_height     =  4.6 # top + bottom + h from generate_map_plot
    #plots.left_margin     =  1.2 # set dynamically below
    plots.bottom_margin   =  0.8
    plots.h               =  3.6
    plots.w               =  3.875
    plots.cbar_width      =  0.25
    plots.cbar_gap        =  0.6
    plots.density         =  100
    plots.n_cols          =  2
    plots.contour_labels  = False
    plots.cbar_title      = True
    plots.ylabelpad       = -1.0
    plots.xlabelpad       =  0.5
    plots.clabelpad       =  4.0
    plots.scatter_size    =  0.0
    plots.x_major_ticks   = 0.5
    plots.x_minor         = 5
    plots.y_major_ticks   = 1.0
    plots.y_minor         = 5
    plots.draw_cell_box   = False

    zeta = zeta_table['zeta [MHz]']
    w2 = zeta_table['w2 [GHz]']
    wc = zeta_table['wc [GHz]']

    plots.add_cell(w2, wc, np.abs(zeta), title=r'$\zeta$~(MHz)', logscale=True,
                   left_margin=1.2, y_labels=True)

    gamma = -2.0 * np.pi * (zeta/1000.0) * T
    C = np.abs(np.sin(0.5*gamma))
    plots.add_cell(w2, wc, C, vmin=0.0, vmax=1.0, title='concurrence at $T = 50$~ns',
                   left_margin=0.7, y_labels=False)

    if OUTFOLDER is not None:
        outfile = os.path.join(OUTFOLDER, outfile)

    fig = plots.plot(quiet=False, show=False, style=STYLE)
    fig.savefig(outfile)
    print("written %s" % outfile)
    plt.close(fig)


@FuncFormatter
def weyl_x_tick_fmt(x, pos):
    'The two args are the value and tick position'
    if x == 0:
        return '0'
    elif x == 1:
        return '1'
    else:
        return ''
        #return ("%.1f" % x)[1:]

@FuncFormatter
def weyl_y_tick_fmt(y, pos):
    'The two args are the value and tick position'
    if y == 0:
        return '0'
    elif y == 0.5:
        return '0.5'
    else:
        return ''


@FuncFormatter
def weyl_z_tick_fmt(z, pos):
    'The two args are the value and tick position'
    if z == 0.5:
        return '0.5'
    else:
        return ''


def generate_map_plot(stage_table_200, stage_table_050, stage_table_010,
        outfile):

    # axes:
    # * 200
    # * 50
    # * 10

    # vertical layout parameters
    top = 0.2     # top of figure to T=200 axes
    bottom = 0.8  # bottom of figure to T=010 axes
    gap = 0.4     # vertical gap between axes
    h = 3.6       # height of all axes
    cell_height = ((bottom + top + 2*gap) / 3.0) + h # 4.2
    bottom_margin = { # within each cell
         10: bottom,
         50: (bottom + h + gap) - cell_height,
        200: (bottom + 2*(h+gap)) - 2*cell_height,
    }

    weyl_bottom_offset = {
         10: bottom,
         50: cell_height + bottom_margin[50],
        200: 2*cell_height + bottom_margin[200],
    }

    weyl_label_offset = {
         10: bottom + h - 0.25,
         50: cell_height + bottom_margin[50] + h - 0.25,
        200: 2*cell_height + bottom_margin[200] + h -  0.25,
    }

    # set up map plot
    map_plots = PlotGrid(layout='paper')
    map_plots.cell_width      =  6.375
    map_plots.cell_height     =  cell_height
    #map_plots.left_margin     =  1.2 # set dynamically below
    #map_plots.bottom_margin   =  1.0 # set dynamically below
    map_plots.h               =  h
    map_plots.w               =  3.875
    map_plots.cbar_width      =  0.25
    map_plots.cbar_gap        =  0.6
    map_plots.density         =  100
    map_plots.n_cols          =  2
    map_plots.contour_labels  = False
    map_plots.cbar_title      = True
    map_plots.ylabelpad       = -1.0
    map_plots.xlabelpad       =  0.5
    map_plots.clabelpad       =  4.0
    map_plots.scatter_size    =  0.0
    map_plots.x_major_ticks   = 0.5
    map_plots.x_minor         = 5
    map_plots.y_major_ticks   = 1.0
    map_plots.y_minor         = 5
    map_plots.draw_cell_box   = False

    # set up Weyl plot
    weyl_fig_width = 4.35
    weyl_fig_height = 3 * cell_height
    fig_weyl = new_figure(weyl_fig_width, weyl_fig_height, style=STYLE)
    weyl_left_margin = 0.3
    weyl_w = 3.8
    weyl_h = 3.2
    ax_weyl = {}

    data = OrderedDict([
            (200, stage_table_200),
            (50,  stage_table_050),
            (10,  stage_table_010), ])

    for T in data.keys():

        stage_table = data[T]
        min_err = diss_error(gamma=1.2e-5, t=T)

        # filter stage table to single frequencies
        stage_table = stage_table[stage_table['category'].str.contains('1freq')]

        # get quality table (best of any entry for a given parameter point)
        (__, t_PE), (__, t_SQ) = stage_table.groupby('target', sort=True)
        Q_table = get_Q_table(t_PE, t_SQ)
        Q_table = Q_table\
                .groupby(['w1 [GHz]', 'w2 [GHz]', 'wc [GHz]'], as_index=False)\
                .apply(lambda df: df.sort('Q').tail(1))\
                .reset_index(level=0, drop=True)

        # get concurrence table (best of any antry for a given parameter point)
        C_table = t_PE\
                .groupby(['w1 [GHz]', 'w2 [GHz]', 'wc [GHz]'], as_index=False)\
                .apply(lambda df: df.sort('J_PE').head(1))\
                .reset_index(level=0, drop=True)

        # plot the maps
        map_x_labels = False
        if T == 10:
            map_x_labels = True

        map_plots.add_cell(C_table['w2 [GHz]'], C_table['wc [GHz]'],
                    C_table['C'], vmin=0.0, vmax=1.0,
                    val_alpha=(1-t_PE['max loss']), bg='black',
                    contour_levels=0, logscale=False, title='concurrence',
                    left_margin=1.2, bottom_margin=bottom_margin[T],
                    x_labels=map_x_labels, y_labels=True)
        map_plots.add_cell(Q_table['w2 [GHz]'], Q_table['wc [GHz]'],
                       1.0-Q_table['Q'], vmin=min_err, vmax=1.0, logscale=True,
                        val_alpha=(1-t_PE['max loss']), bg='black',
                        cmap=plt.cm.gnuplot2_r,
                        contour_levels=0, title='$1-Q$',
                        left_margin=0.7, bottom_margin=bottom_margin[T],
                        x_labels=map_x_labels, y_labels=False)

        # plot the weyl_chamber
        pos_weyl = [weyl_left_margin/weyl_fig_width,
                    (weyl_bottom_offset[T])/weyl_fig_height,
                    weyl_w/weyl_fig_width, weyl_h/weyl_fig_height]
        ax_weyl = fig_weyl.add_axes(pos_weyl, projection='3d')
        w = QDYN.weyl.WeylChamber()
        t_PE_weyl = t_PE[(t_PE['max loss']<0.1) & (t_PE['C']==1.0)]
        w.scatter(t_PE_weyl['c1'], t_PE_weyl['c2'], t_PE_weyl['c3'],
                  s=5, linewidth=0)
        w.render(ax_weyl)
        ax_weyl.xaxis._axinfo['ticklabel']['space_factor'] = 1.0
        ax_weyl.yaxis._axinfo['ticklabel']['space_factor'] = 1.0
        ax_weyl.zaxis._axinfo['ticklabel']['space_factor'] = 1.3
        ax_weyl.xaxis._axinfo['label']['space_factor'] = 1.5
        ax_weyl.yaxis._axinfo['label']['space_factor'] = 1.5
        ax_weyl.zaxis._axinfo['label']['space_factor'] = 1.5
        ax_weyl.xaxis.set_major_formatter(weyl_x_tick_fmt)
        ax_weyl.yaxis.set_major_formatter(weyl_y_tick_fmt)
        ax_weyl.zaxis.set_major_formatter(weyl_z_tick_fmt)
        fig_weyl.text(0.5, weyl_label_offset[T]/weyl_fig_height,
                      r'$T = %d$~ns' % T, verticalalignment='top',
                      horizontalalignment='center', size=10)

    if OUTFOLDER is not None:
        outfile = os.path.join(OUTFOLDER, outfile)

    out_path, out_filename = os.path.split(outfile)
    out_base, out_ext = os.path.splitext(out_filename)
    outfile_maps = os.path.join(out_path, "%s_map%s" % (out_base, out_ext))
    outfile_weyl = os.path.join(out_path, "%s_weyl%s" % (out_base, out_ext))

    fig_maps = map_plots.plot(quiet=False, show=False, style=STYLE)
    fig_maps.savefig(outfile_maps)
    print("written %s" % outfile_maps)
    plt.close(fig_maps)

    fig_weyl.savefig(outfile_weyl)
    print("written %s" % outfile_weyl)
    plt.close(fig_weyl)


def generate_weyl_plot(stage_table, outfile):
    from QDYN.weyl import WeylChamber
    w = QDYN.weyl.WeylChamber()
    w.fig_width = 5.5
    w.fig_height = 4.1
    w.left_margin = 0.3
    w.right_margin = 0.0
    w.dpi = 600
    w.ticklabelsize = 7.5
    (__, t_PE), (__, __) = stage_table\
         [stage_table['category'].str.contains('1freq')]\
         .groupby('target', sort=True)
    t_PE = t_PE[(t_PE['max loss']<0.1) & (t_PE['C']==1.0)]
    w.scatter(t_PE['c1'], t_PE['c2'], t_PE['c3'], s=5, linewidth=0)
    if OUTFOLDER is not None:
        outfile = os.path.join(OUTFOLDER, outfile)
    fig = new_figure(w.fig_width, w.fig_height, style=STYLE)
    w.plot(fig=fig, outfile=outfile)
    print("written %s" % outfile)


def generate_popdyn_plot(outfile):
    dyn = QDYNTransmonLib.popdyn.PopPlot(
          "./propagate/010_RWA_w2_6000MHz_wc_6300MHz_stage3/PE_1freq/",
          panel_width=3.5, left_margin=1.5, right_margin=3.0, top_margin=0.7)
    dyn.styles['tot']['ls'] = '--'
    dyn.plot(pops=('00', '01', '10', '11', 'tot'), in_panel_legend=False)
    if OUTFOLDER is not None:
        outfile = os.path.join(OUTFOLDER, outfile)
    plt.savefig(outfile)
    print("written %s" % outfile)


def generate_error_plot(outfile):

    fig_height      = 4.0
    fig_width       = 8.5               # Total canvas (cv) width
    left_margin     = 1.2               # Left cv -> plot area
    right_margin    = 0.2               # plot area -> right cv
    top_margin      = 0.25              # top cv -> plot area
    bottom_margin   = 0.6
    h = fig_height - (bottom_margin + top_margin)
    w = fig_width  - (left_margin + right_margin)
    data = r'''
    #                   minimum error  achieved PE error  achieved quality error
    # gate duration [ns]
    5                        3.77e-04           1.54e-03                9.83e-04
    10                       7.54e-04           8.84e-04                8.22e-04
    20                       1.51e-03           1.56e-03                1.54e-03
    50                       3.76e-03           3.87e-03                3.82e-03
    100                      7.51e-03           7.59e-03                7.56e-03
    200                      1.49e-02           1.50e-02                1.50e-02
    '''
    T, eps_0, eps_PE, eps_Q = np.genfromtxt(StringIO(data), unpack=True)
    fig = new_figure(fig_width, fig_height, style=STYLE)
    pos = [left_margin/fig_width, bottom_margin/fig_height,
           w/fig_width, h/fig_height]
    ax = fig.add_axes(pos)
    ax.plot(T, eps_0, label=r'$\varepsilon_{\text{avg}}^0$', marker='o', ls='dotted')
    ax.plot(T, eps_PE, label=r'$\varepsilon_{\text{avg}}^{\text{PE}}$', marker='o', ls='dashed')
    ax.plot(T, eps_Q, label=r'$\varepsilon_{\text{avg}}^{\text{Q}}$', marker='o', ls='solid')
    ax.legend(loc='lower right')
    ax.annotate('QSL', xy=(10, 1e-3),  xycoords='data',
                xytext=(10, 1e-2), textcoords='data',
                arrowprops=dict(facecolor='black', width=1, headwidth=3, shrink=0.05),
                horizontalalignment='center', verticalalignment='top',
                )
    set_axis(ax, 'x', 4, 210, label='gate time (ns)', logscale=True, labelpad=-2)
    set_axis(ax, 'y', 1e-4, 1.0e-1, label='lowest gate error', logscale=True)
    ax.tick_params(axis='x', pad=3)

    if OUTFOLDER is not None:
        outfile = os.path.join(OUTFOLDER, outfile)
    fig.savefig(outfile, format=os.path.splitext(outfile)[1][1:])
    print("written %s" % outfile)


def main(argv=None):

    if argv is None:
        argv = sys.argv
    if not os.path.isdir(OUTFOLDER):
        QDYN.shutil.mkdir(OUTFOLDER)

    stage_table_200 = get_stage3_table('./runs_200_RWA')
    stage_table_050 = get_stage3_table('./runs_020_RWA')
    stage_table_010 = get_stage3_table('./runs_010_RWA')
    zeta_table = get_zeta_table('./runs_050_RWA', T=50)

    # Fig 1
    generate_field_free_plot(zeta_table, T=50, outfile='fig1.pdf')
    # Fig 2
    generate_map_plot(stage_table_200, stage_table_050, stage_table_010,
                      outfile='fig2.pdf')
    # Fig 3
    generate_error_plot(outfile='fig3.pdf')
    # Fig 5
    #generate_popdyn_plot(outfile='popdyn.png')


if __name__ == "__main__":
    sys.exit(main())
