#!/usr/bin/env python
import os
import sys
from StringIO import StringIO
import QDYN
import QDYNTransmonLib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
from notebook_utils import get_Q_table, diss_error, PlotGrid
from notebook_utils import get_stage1_table, get_stage2_table, get_stage3_table
from mgplottools.mpl import new_figure, set_axis

STYLE = 'slides.mplstyle'

get_stage1_table = QDYN.memoize.memoize(get_stage1_table)
get_stage1_table.load('stage1_table.cache')
get_stage2_table = QDYN.memoize.memoize(get_stage2_table)
get_stage2_table.load('stage2_table.cache')
get_stage3_table = QDYN.memoize.memoize(get_stage3_table)
get_stage3_table.load('stage3_table.cache')

OUTFOLDER = './ibm_images'


def generate_field_free_plot(stage_table, T, outfile):
    """Plot field-free concurrence"""
    plots = PlotGrid(layout='poster')
    plots.n_cols = 1
    plots.scatter_size = 0
    table = stage_table[stage_table['category']=='field_free']
    plots.add_cell(table['w2 [GHz]'], table['wc [GHz]'], table['C'],
                   vmin=0.0, vmax=1.0, title='concurrence')
    fig = plots.plot(quiet=True, show=False, style=STYLE)
    if OUTFOLDER is not None:
        outfile = os.path.join(OUTFOLDER, outfile)
    fig.savefig(outfile)
    print("written %s" % outfile)
    plt.close(fig)


def generate_map_plot(stage_table, T, outfile):

    # set up plot
    plots = PlotGrid(layout='poster')
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

    # plotting
    plots.add_cell(C_table['w2 [GHz]'], C_table['wc [GHz]'],
                   C_table['C'], vmin=0.0, vmax=1.0,
                   contour_levels=0, logscale=False, title='concurrence')
    plots.add_cell(Q_table['w2 [GHz]'], Q_table['wc [GHz]'], 1.0-Q_table['Q'],
                    vmin=min_err, vmax=1.0, logscale=True,
                    cmap=plt.cm.gnuplot2_r,
                    contour_levels=0, title='$1-Q$')

    fig = plots.plot(quiet=True, show=False, style=STYLE)
    if OUTFOLDER is not None:
        outfile = os.path.join(OUTFOLDER, outfile)
    fig.savefig(outfile)
    print("written %s" % outfile)
    plt.close(fig)


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

    fig_height      =  7.5*0.666
    fig_width       = 10.5*0.666        # Total canvas (cv) width
    left_margin     = 1.1*0.666         # Left cv -> plot area
    right_margin    = 0.4*0.666         # plot area -> right cv
    top_margin      = 0.4*0.666         # top cv -> plot area
    bottom_margin   = 1.3*0.666         # bottom cv -> plot area
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
    ax.plot(T, eps_0, label=r'$\varepsilon_{\text{avg}}^0$', marker='o')
    ax.plot(T, eps_PE, label=r'$\varepsilon_{\text{avg}}^{\text{PE}}$', marker='o')
    ax.plot(T, eps_Q, label=r'$\varepsilon_{\text{avg}}^{\text{Q}}$', marker='o')
    ax.legend(loc='lower right')
    ax.annotate('QSL', xy=(10, 1e-3),  xycoords='data',
                xytext=(10, 1e-2), textcoords='data',
                arrowprops=dict(facecolor='black', width=1, headwidth=3, shrink=0.05),
                horizontalalignment='center', verticalalignment='top',
                )
    set_axis(ax, 'x', 4, 210, label='gate time (ns)', logscale=True)
    set_axis(ax, 'y', 1e-4, 1.0e-1, label='lowest gate error', logscale=True)
    if OUTFOLDER is not None:
        outfile = os.path.join(OUTFOLDER, outfile)
    fig.savefig(outfile, format=os.path.splitext(outfile)[1][1:])
    print("written %s" % outfile)


def main(argv=None):
    if argv is None:
        argv = sys.argv
    generate_popdyn_plot(outfile='popdyn.png')
    generate_error_plot(outfile='qsl.pdf')
    generate_field_free_plot(get_stage1_table('./runs_050_RWA'), T=50, outfile='field_free_050.png')
    generate_map_plot(get_stage3_table('./runs_200_RWA'), T=200, outfile='map_200.png')
    generate_map_plot(get_stage3_table('./runs_100_RWA'), T=100, outfile='map_100.png')
    generate_map_plot(get_stage3_table('./runs_050_RWA'), T=50,  outfile='map_050.png')
    generate_map_plot(get_stage3_table('./runs_020_RWA'), T=20,  outfile='map_020.png')
    generate_map_plot(get_stage3_table('./runs_010_RWA'), T=10,  outfile='map_010.png')
    generate_map_plot(get_stage3_table('./runs_005_RWA'), T=10,  outfile='map_005.png')
    generate_weyl_plot(get_stage3_table('./runs_200_RWA'), outfile='weyl_200.pdf')
    generate_weyl_plot(get_stage3_table('./runs_100_RWA'), outfile='weyl_100.pdf')
    generate_weyl_plot(get_stage3_table('./runs_050_RWA'), outfile='weyl_050.pdf')
    generate_weyl_plot(get_stage3_table('./runs_020_RWA'), outfile='weyl_020.pdf')
    generate_weyl_plot(get_stage3_table('./runs_010_RWA'), outfile='weyl_010.pdf')
    generate_weyl_plot(get_stage3_table('./runs_005_RWA'), outfile='weyl_005.pdf')



if __name__ == "__main__":
    sys.exit(main())
