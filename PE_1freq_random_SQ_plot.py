#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for generating a plot of gate error (data included in file)
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.mlab import griddata
from matplotlib.colors import LogNorm
from notebook_utils import get_selection_plot_data, get_selection_quality
import matplotlib.pylab as plt
from select_for_stage2 import all_select_runs
from QDYN.memoize import memoize
from mgplottools.mpl import get_color, set_axis, new_figure, ls, \
                            set_color_cycle

STYLE = 'PE_1freq_random_plot.mplstyle'


def render_values(w_2, w_c, val, fig, ax_contour, ax_cbar, density=100,
    logscale=False, vmin=None, vmax=None, n_contours=10, y_axis_label=True,
    colorlabel='concurrence'):
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
                                    vmax=vmax, vmin=vmin)
    ax_contour.scatter(w_2, w_c, marker='o', c='cyan', s=2, zorder=10, lw=0.3)
    set_axis(ax_contour, 'x', 6.0, 7.5, 0.5, range=(6.1, 7.5), minor=5)
    if y_axis_label:
        set_axis(ax_contour, 'y', 5.0, 11.1, 1.0, minor=5)
        ax_contour.set_ylabel(r"$\omega_c$ (GHz)", labelpad=-1.0)
    else:
        set_axis(ax_contour, 'y', 5.0, 11.1, 1.0, minor=5, ticklabels=False)
    ax_contour.set_xlabel(r"$\omega_2$ (GHz)", labelpad=-0.3)
    # show the resonance line (cavity on resonace with qubit 2)
    ax_contour.plot(np.linspace(5.0, 11.1, 10), np.linspace(5.0, 11.1, 10), color='white', lw=1.0)
    ax_contour.plot(np.linspace(6.0, 7.5, 10), 6.0*np.ones(10), color='white', lw=1.0)
    ax_contour.axvline(6.29, color='white', ls='--')
    ax_contour.axvline(6.31, color='white', ls='--')
    #ax_contour.axvline(6.58, color='white', ls='--')
    #ax_contour.axvline(6.62, color='white', ls='--')
    axT = ax_cbar.twinx()
    fig.colorbar(contours, cax=axT)
    ax_cbar.set_yticks([])
    ax_cbar.set_yticklabels('',visible=False)
    ax_cbar.set_ylabel(colorlabel, labelpad=1.0)


def create_figure(outfile, select_data):

    plot_width      =  5.5
    left_margin     =  0.9
    top_margin      =  0.2
    bottom_margin   =  0.7
    h               =  3.5
    w               =  3.3
    h_offset        =  5.2
    cbar_width      =  0.20
    cbar_gap        =  0.5

    target = 'PE'

    fig_width = 2*plot_width
    fig_height = bottom_margin + h + top_margin

    fig = new_figure(fig_width, fig_height, style=STYLE)
    from matplotlib import rcParams
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'

    best_C = None # best from all categories
    best_loss = None # best from all categories
    best_J = None

    def J(C, loss):
        if target == 'PE':
            return 1.0 - C + loss
        elif target == 'SQ':
            return C + loss

    target_category = 'SQ_1freq_random'
    w_2, w_c, C, loss = get_selection_plot_data(select_data,
                                                target_category)
    __, __, quality = get_selection_quality(select_data)['1freq_random']
    # collect the "best" result over all pulse categories
    if best_C is None:
        best_C = C.copy()
        best_loss = loss.copy()
        best_J = np.zeros(len(C))
        for i in xrange(len(C)):
            best_J[i] = J(C[i], loss[i])
    else:
        for i in xrange(len(C)):
            if J(C[i], loss[i]) < best_J[i]:
                best_C[i] = C[i]
                best_loss[i] = loss[i]
                best_J[i] = J(C[i], loss[i])

    pos_contour = [left_margin/fig_width,
                (bottom_margin)/fig_height,
                w/fig_width, h/fig_height]
    ax_contour = fig.add_axes(pos_contour)
    pos_cbar = [(left_margin+w+cbar_gap)/fig_width,
                (bottom_margin)/fig_height,
                cbar_width/fig_width, h/fig_height]
    ax_cbar = fig.add_axes(pos_cbar)
    render_values(w_2, w_c, quality, fig, ax_contour, ax_cbar, vmin=0.0,
                vmax=1.0, colorlabel='comb. 1Q/2Q-gate quality')

    pos_contour = [(left_margin+h_offset)/fig_width,
                (bottom_margin)/fig_height,
                w/fig_width, h/fig_height]
    ax_contour = fig.add_axes(pos_contour)
    pos_cbar = [(left_margin+w+cbar_gap+h_offset)/fig_width,
                (bottom_margin)/fig_height,
                cbar_width/fig_width, h/fig_height]
    ax_cbar = fig.add_axes(pos_cbar)
    render_values(w_2, w_c, 1-C, fig, ax_contour, ax_cbar,
                    logscale=False, vmin=0.0, vmax=1.0, y_axis_label=False,
                    colorlabel='disentanglement: 1-C')

    # output
    fig.savefig(outfile, format=os.path.splitext(outfile)[1][1:])


@memoize
def read_data():
    return all_select_runs()



def main(argv=None):
    if argv is None:
        argv = sys.argv
    basename = os.path.splitext(__file__)[0]
    outfile = basename + '.png'
    read_data.load('PE_1freq_random_plot.cache')
    select_data = read_data()
    read_data.dump('PE_1freq_random_plot.cache')
    create_figure(outfile, select_data)


if __name__ == "__main__":
    sys.exit(main())

