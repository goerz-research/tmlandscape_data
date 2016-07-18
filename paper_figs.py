#!/usr/bin/env python
import os
from os.path import join
import sys
from StringIO import StringIO
import QDYN
import QDYNTransmonLib
import numpy as np
import matplotlib
from collections import OrderedDict
matplotlib.use('Agg')
import matplotlib.pylab as plt
from notebook_utils import diss_error, render_values
from notebook_utils import get_stage3_table, get_zeta_table, read_target_gate
from mgplottools.mpl import new_figure, set_axis, get_color, ls
from matplotlib.ticker import FuncFormatter
import pandas as pd

"""
This script generates all the plots for the paper. All the required data a part
of the repository. That is, running

    ./create_venv.sh
    . setenv.sh
    ./paper_figs.py

from a clean checkout should be sufficient to generate the figures.
"""

STYLE = 'paper.mplstyle'

get_stage3_table = QDYN.memoize.memoize(get_stage3_table)
get_stage3_table.load('stage3_table.cache')
get_zeta_table = QDYN.memoize.memoize(get_zeta_table)
get_zeta_table.load('zeta_table.cache')

OUTFOLDER = './paper_images'
#OUTFOLDER = '/Users/goerz/Documents/Papers/TransmonLandscape'


def generate_field_free_plot(zeta_table, T, outfile):
    """Plot field-free entangling energy zeta, and projected concurrence after
    the given gate duration T in ns.
    """
    left_margin   = 1.05
    cbar_width    = 0.25
    cbar_gap      = 0.25
    hgap1         = 1.13
    hgap2         = 0.93
    right_margin  = 1.0
    w             = 4.2

    top_margin    = 0.7
    bottom_margin = 0.8
    h             = 2.5

    density = 300
    wc_min = 4.5
    wc_max = 9.0
    w2_min = 5.0
    w2_max = 7.0
    w1 = 6.0
    g = 0.07
    alpha = 0.3

    y_tick0 = -3
    y_tick1 =  3
    y_major_ticks = 1
    y_minor = 2
    x_tick0 =  -20
    x_tick1 =  40
    x_major_ticks = 10
    x_minor = 5
    xlabelpad = 3.0
    ylabelpad = 1.0

    Delta2 = lambda w2: (w2 - w1)/alpha
    DeltaC = lambda wc: (wc - w1)/g
    y_range = (Delta2(w2_min), Delta2(w2_max))
    x_range = (DeltaC(wc_min), DeltaC(wc_max))

    fig_height = bottom_margin + top_margin + h
    fig_width  = (left_margin + 2 * cbar_gap + 3 * cbar_width +
                  hgap1 + hgap2 + right_margin + 3*w)

    fig = new_figure(fig_width, fig_height, style=STYLE)
    axs = []
    cbar_axs = []

    # Zeta

    zeta = zeta_table['zeta [MHz]']
    abs_zeta = np.clip(np.abs(zeta), a_min=1e-5, a_max=1e5)
    w2 = zeta_table['w2 [GHz]']
    wc = zeta_table['wc [GHz]']

    pos = [left_margin/fig_width, bottom_margin/fig_height,
           w/fig_width, h/fig_height]
    ax = fig.add_axes(pos); axs.append(ax)
    pos_cbar = [(left_margin+w+cbar_gap)/fig_width, bottom_margin/fig_height,
               cbar_width/fig_width, h/fig_height]
    ax_cbar = fig.add_axes(pos_cbar); cbar_axs.append(ax_cbar)
    cbar = render_values(wc, w2, abs_zeta, ax, ax_cbar, density=density,
                         logscale=True, vmin=1e-1,
                         transform_x=DeltaC, transform_y=Delta2)
    cbar.ax.yaxis.set_ticks(cbar.norm(np.concatenate(
            [np.arange(0.1, 1, 0.1), np.arange(1, 10, 1),
            np.arange(10, 100, 10)])), minor=True)
    set_axis(ax, 'y', y_tick0, y_tick1, y_major_ticks,
             range=y_range, minor=y_minor)
    set_axis(ax, 'x', x_tick0, x_tick1, x_major_ticks,
             range=x_range, minor=x_minor)
    ax.tick_params(which='both', direction='out')
    ax.set_ylabel(r"$\Delta_2/\alpha$", labelpad=ylabelpad)
    ax.set_xlabel(r"$\Delta_c/g$", labelpad=xlabelpad)
    fig.text((left_margin + w + cbar_gap + cbar_width+0.53)/fig_width,
              1-0.2/fig_height, r'$\zeta$~(MHz)', verticalalignment='top',
              horizontalalignment='right')
    labels = [
    #          w_c   w_2     label pos
        ("A", (5.75, 6.32 ), (5.35, 6.40), 'FireBrick'),
        ("B", (6.20, 5.90 ), (6.35, 5.95), 'OrangeRed')
    ]
    for (label, x_y_data, x_y_label, color) in labels:
        ax.scatter((DeltaC(x_y_data[0]),), (Delta2(x_y_data[1]), ),
                   color=color, marker='x')
        ax.annotate(label, (DeltaC(x_y_label[0]), Delta2(x_y_label[1])),
                    color=color)

    # Entanglement time

    T_entangling = 500.0/abs_zeta

    pos = [(left_margin+cbar_gap+cbar_width+hgap1+w)/fig_width,
            bottom_margin/fig_height, w/fig_width, h/fig_height]
    ax = fig.add_axes(pos)
    pos_cbar = [(left_margin+2*(w+cbar_gap)+hgap1+cbar_width)/fig_width,
                bottom_margin/fig_height, cbar_width/fig_width, h/fig_height]
    ax_cbar = fig.add_axes(pos_cbar)
    cbar = render_values(wc, w2, T_entangling, ax, ax_cbar, density=density,
                         logscale=True, vmax=1e3,
                         transform_x=DeltaC, transform_y=Delta2)
    cbar.ax.yaxis.set_ticks(cbar.norm(np.concatenate(
            [np.arange(4, 10, 1), np.arange(10, 100, 10),
            np.arange(100, 1000, 100)])), minor=True)
    set_axis(ax, 'y', y_tick0, y_tick1, y_major_ticks, range=y_range,
             minor=y_minor, ticklabels=False)
    set_axis(ax, 'x', x_tick0, x_tick1, x_major_ticks, range=x_range,
             minor=x_minor)
    ax.tick_params(which='both', direction='out')
    ax.set_xlabel(r"$\Delta_c/g$", labelpad=xlabelpad)
    fig.text((left_margin + 2*(w + cbar_gap + cbar_width)+hgap1+0.53)/fig_width,
              1-0.2/fig_height, r'$T(C_0=1)$ (ns)', verticalalignment='top',
              horizontalalignment='right')
    labels = [
    #          w_c   w_2     label pos
        ("A", (5.75, 6.32 ), (5.35, 6.40), 'FireBrick'),
        ("B", (6.20, 5.90 ), (6.35, 5.95), 'FireBrick')
    ]
    for (label, x_y_data, x_y_label, color) in labels:
        ax.scatter((DeltaC(x_y_data[0]),), (Delta2(x_y_data[1]), ),
                   color=color, marker='x')
        ax.annotate(label, (DeltaC(x_y_label[0]), Delta2(x_y_label[1])),
                    color=color)

    # Relative effective decay rate

    gamma_bare = 0.012
    rel_decay = zeta_table['gamma [MHz]'] / gamma_bare
    print("Min: %s" % np.min(rel_decay))
    print("Max: %s" % np.max(rel_decay))

    pos = [(left_margin+w+cbar_gap+cbar_width+hgap1+w
                 +cbar_gap+cbar_width+hgap2)/fig_width,
            bottom_margin/fig_height, w/fig_width, h/fig_height]
    ax = fig.add_axes(pos)
    pos_cbar = [(left_margin+w+cbar_gap+cbar_width+hgap1+w
                 +cbar_gap+cbar_width+hgap2+w+cbar_gap)/fig_width,
                bottom_margin/fig_height, cbar_width/fig_width, h/fig_height]
    ax_cbar = fig.add_axes(pos_cbar)
    cbar = render_values(wc, w2, rel_decay, ax, ax_cbar, density=density,
                         logscale=False, vmin=1, vmax=2.3,
                         cmap=plt.cm.cubehelix_r,
                         transform_x=DeltaC, transform_y=Delta2)
    cbar.set_ticks([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2])
    set_axis(ax, 'y', y_tick0, y_tick1, y_major_ticks, range=y_range, minor=y_minor,
             ticklabels=False)
    set_axis(ax, 'x', x_tick0, x_tick1, x_major_ticks, range=x_range, minor=x_minor)
    ax.tick_params(which='both', direction='out')
    ax.set_xlabel(r"$\Delta_c/g$", labelpad=xlabelpad)
    fig.text(0.995, 1-0.2/fig_height,
             r'$\gamma_{\text{dressed}} / \gamma_{\text{bare}}$',
             verticalalignment='top', horizontalalignment='right')
    labels = [
    #          w_c   w_2     label pos
        ("A", (5.75, 6.32 ), (5.35, 6.40), 'FireBrick'),
        ("B", (6.20, 5.90 ), (6.35, 5.95), 'FireBrick')
    ]
    for (label, x_y_data, x_y_label, color) in labels:
        ax.scatter((DeltaC(x_y_data[0]),), (Delta2(x_y_data[1]), ),
                   color=color, marker='x')
        ax.annotate(label, (DeltaC(x_y_label[0]), Delta2(x_y_label[1])),
                    color=color)

    if OUTFOLDER is not None:
        outfile = os.path.join(OUTFOLDER, outfile)

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


def generate_map_plot_SQ(stage_table_200, stage_table_050, stage_table_010,
        zeta_table, outfile):

    left_margin   = 1.1
    hgap          = 0.35
    cbar_width    = 0.25
    cbar_gap      = hgap
    right_margin  = 1.0
    w             = 4.2

    top_margin    = 0.3
    bottom_margin = 0.8
    vgap          = 0.45
    h             = 2.5

    density = 300
    wc_min = 4.5
    wc_max = 9.0
    w2_min = 5.0
    w2_max = 7.0
    w1 = 6.0
    g = 0.07
    alpha = 0.3

    y_tick0 = -3
    y_tick1 =  3
    y_major_ticks = 1
    y_minor = 2
    x_tick0 =  -20
    x_tick1 =  40
    x_major_ticks = 10
    x_minor = 5
    xlabelpad = 3.0
    ylabelpad = 1.0

    Delta2 = lambda w2: (w2 - w1)/alpha
    DeltaC = lambda wc: (wc - w1)/g
    y_range = (Delta2(w2_min), Delta2(w2_max))
    x_range = (DeltaC(wc_min), DeltaC(wc_max))

    fig_height = bottom_margin + 3*h + 2*vgap + top_margin
    fig_width  = (left_margin + 3*w + 2*hgap + cbar_gap + cbar_width
                  + right_margin)
    fig = new_figure(fig_width, fig_height, style=STYLE)

    data = OrderedDict([
            (200, stage_table_200),
            (50,  stage_table_050),
            (10,  stage_table_010), ])


    for i_col, T in enumerate(data.keys()):

        stage_table = data[T]
        min_err = diss_error(gamma=1.2e-5, t=T)

        # filter stage table to single frequencies
        stage_table = stage_table[stage_table['category'].str.contains('1freq')]

        # get optimized concurrence table
        (__, t_PE), (__, t_SQ) = stage_table.groupby('target', sort=True)
        C_opt_table = t_SQ\
                .groupby(['w1 [GHz]', 'w2 [GHz]', 'wc [GHz]'], as_index=False)\
                .apply(lambda df: df.sort('J_PE').head(1))\
                .reset_index(level=0, drop=True)

        zeta = zeta_table['zeta [MHz]']
        gamma = -2.0 * np.pi * (zeta/1000.0) * T # entangling phase
        C_ff = np.abs(np.sin(0.5*gamma))

        # table of zetas at the same data points as C_opt_table
        ind = ['w1 [GHz]', 'w2 [GHz]', 'wc [GHz]']
        zeta_table2 = pd.merge(C_opt_table[ind+['C', 'max loss']],
                              zeta_table[ind+['zeta [MHz]']],
                              on=ind, how='left').dropna()
        zeta2 = zeta_table2['zeta [MHz]']
        gamma2 = -2.0 * np.pi * (zeta2/1000.0) * T # entangling phase
        C_ff2 = np.abs(np.sin(0.5*gamma2))

        # row 1: 1-C_0
        pos = [(left_margin+i_col*(w+hgap))/fig_width,
               (bottom_margin+2*(h+vgap))/fig_height,
               w/fig_width, h/fig_height]
        ax = fig.add_axes(pos);
        if T == 10:
            pos_cbar = [(left_margin+i_col*(w+hgap)+w+cbar_gap)/fig_width,
                        (bottom_margin+2*(h+vgap))/fig_height,
                        cbar_width/fig_width, h/fig_height]
            ax_cbar = fig.add_axes(pos_cbar)
        else:
            ax_cbar = None
        cbar = render_values(zeta_table['wc [GHz]'], zeta_table['w2 [GHz]'],
                1-C_ff, ax, ax_cbar, density=density, vmin=0.0, vmax=1.0,
                transform_x=DeltaC, transform_y=Delta2)
        if ax_cbar is not None:
            ax_cbar.set_ylabel(r'$1-C_0$', rotation=90)
            cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax_cbar.yaxis.set_ticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
        set_axis(ax, 'y', y_tick0, y_tick1, y_major_ticks,
                range=y_range, minor=y_minor)
        set_axis(ax, 'x', x_tick0, x_tick1, x_major_ticks,
                range=x_range, minor=x_minor, ticklabels=False)
        ax.tick_params(which='both', direction='out')
        if i_col > 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(r"$\Delta_2/\alpha$", labelpad=ylabelpad)
        labels = [
        #          w_2   w_c     label pos
            ("A", (5.75, 6.32 ), (5.35, 6.40), 'FireBrick'),
            ("B", (6.20, 5.90 ), (6.35, 5.95), 'FireBrick')
        ]
        for (label, x_y_data, x_y_label, color) in labels:
            ax.scatter((DeltaC(x_y_data[0]),), (Delta2(x_y_data[1]), ),
                    color=color, marker='x')
            ax.annotate(label, (DeltaC(x_y_label[0]), Delta2(x_y_label[1])),
                        color=color)

        # row 2: 1-C_SQ
        pos = [(left_margin+i_col*(w+hgap))/fig_width,
               (bottom_margin+h+vgap)/fig_height,
               w/fig_width, h/fig_height]
        ax = fig.add_axes(pos);
        if T == 10:
            pos_cbar = [(left_margin+i_col*(w+hgap)+w+cbar_gap)/fig_width,
                        (bottom_margin+h+vgap)/fig_height,
                        cbar_width/fig_width, h/fig_height]
            ax_cbar = fig.add_axes(pos_cbar)
        else:
            ax_cbar = None
        cbar = render_values(C_opt_table['wc [GHz]'], C_opt_table['w2 [GHz]'],
                             1-C_opt_table['C'], ax, ax_cbar, density=density,
                             vmin=0.0, vmax=1.0, bg='black',
                             val_alpha=(1-C_opt_table['max loss']),
                             transform_x=DeltaC, transform_y=Delta2)
        if ax_cbar is not None:
            ax_cbar.set_ylabel(r'$1-C_{\text{SQ}}$ (opt)', rotation=90)
            cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax_cbar.yaxis.set_ticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
        set_axis(ax, 'y', y_tick0, y_tick1, y_major_ticks,
                range=y_range, minor=y_minor)
        set_axis(ax, 'x', x_tick0, x_tick1, x_major_ticks,
                range=x_range, minor=x_minor, ticklabels=False)
        ax.tick_params(which='both', direction='out')
        if i_col > 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(r"$\Delta_2/\alpha$", labelpad=ylabelpad)
        labels = [
        #          w_2   w_c     label pos
            ("A", (5.75, 6.32 ), (5.35, 6.40), 'FireBrick'),
            ("B", (6.20, 5.90 ), (6.35, 5.95), 'FireBrick')
        ]
        for (label, x_y_data, x_y_label, color) in labels:
            ax.scatter((DeltaC(x_y_data[0]),), (Delta2(x_y_data[1]), ),
                    color=color, marker='x')
            ax.annotate(label, (DeltaC(x_y_label[0]), Delta2(x_y_label[1])),
                        color=color)

        # row 3: C_0-C_SQ
        pos = [(left_margin+i_col*(w+hgap))/fig_width,
               bottom_margin/fig_height,
               w/fig_width, h/fig_height]
        ax = fig.add_axes(pos);
        if T == 10:
            pos_cbar = [(left_margin+i_col*(w+hgap)+w+cbar_gap)/fig_width,
                        bottom_margin/fig_height,
                        cbar_width/fig_width, h/fig_height]
            ax_cbar = fig.add_axes(pos_cbar)
        else:
            ax_cbar = None
        cbar = render_values(zeta_table2['wc [GHz]'], zeta_table2['w2 [GHz]'],
                             -zeta_table2['C']+C_ff2,
                             ax, ax_cbar, density=density, vmin=0.0, vmax=1.0,
                             val_alpha=(1-zeta_table2['max loss']), bg='black',
                             transform_x=DeltaC, transform_y=Delta2)
        if ax_cbar is not None:
            ax_cbar.set_ylabel(r'$C_{0} - C_{\text{SQ}}$', rotation=90)
            cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax_cbar.yaxis.set_ticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
        set_axis(ax, 'y', y_tick0, y_tick1, y_major_ticks, range=y_range,
                minor=y_minor)
        set_axis(ax, 'x', x_tick0, x_tick1, x_major_ticks, range=x_range,
                minor=x_minor)
        ax.tick_params(which='both', direction='out')
        if i_col > 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(r"$\Delta_2/\alpha$", labelpad=ylabelpad)
        ax.set_xlabel(r"$\Delta_c/g$", labelpad=xlabelpad)
        labels = [
        #          w_2   w_c     label pos
            ("A", (5.75, 6.32 ), (5.35, 6.40), 'OrangeRed'),
            ("B", (6.20, 5.90 ), (6.35, 5.95), 'OrangeRed')
        ]
        for (label, x_y_data, x_y_label, color) in labels:
            ax.scatter((DeltaC(x_y_data[0]),), (Delta2(x_y_data[1]), ),
                    color=color, marker='x')
            ax.annotate(label, (DeltaC(x_y_label[0]), Delta2(x_y_label[1])),
                        color=color)

        fig.text((left_margin+i_col*(w+hgap)+0.95*w)/fig_width,
                 (bottom_margin+2*(h+vgap)+h-0.2)/fig_height,
                 r'$T = %d$~ns' % T, verticalalignment='top',
                 horizontalalignment='right', size=10)

    if OUTFOLDER is not None:
        outfile = os.path.join(OUTFOLDER, outfile)

    fig.savefig(outfile)
    print("written %s" % outfile)
    plt.close(fig)


def generate_map_plot_PE(stage_table_200, stage_table_050, stage_table_010,
        zeta_table, outfile):

    left_margin   = 1.1
    hgap          = 0.35
    cbar_width    = 0.25
    cbar_gap      = hgap
    right_margin  = 1.0
    w             = 4.2

    top_margin    = 0.0
    bottom_margin = 0.8
    vgap          = 0.45
    h             = 2.5

    weyl_offset_x = 0.3
    weyl_offset_y = -0.1
    weyl_width    = 3.5
    weyl_height   = 2.5

    density = 300
    wc_min = 4.5
    wc_max = 9.0
    w2_min = 5.0
    w2_max = 7.0
    w1 = 6.0
    g = 0.07
    alpha = 0.3

    y_tick0 = -3
    y_tick1 =  3
    y_major_ticks = 1
    y_minor = 2
    x_tick0 =  -20
    x_tick1 =  40
    x_major_ticks = 10
    x_minor = 5
    xlabelpad = 3.0
    ylabelpad = 1.0

    Delta2 = lambda w2: (w2 - w1)/alpha
    DeltaC = lambda wc: (wc - w1)/g
    y_range = (Delta2(w2_min), Delta2(w2_max))
    x_range = (DeltaC(wc_min), DeltaC(wc_max))

    fig_height = bottom_margin + 3*h + 2*vgap + top_margin + 0.5
    fig_width  = (left_margin + 3*w + 2*hgap + cbar_gap + cbar_width
                  + right_margin)
    fig = new_figure(fig_width, fig_height, style=STYLE)

    data = OrderedDict([
            (200, stage_table_200),
            (50,  stage_table_050),
            (10,  stage_table_010), ])


    for i_col, T in enumerate(data.keys()):

        stage_table = data[T]
        min_err = diss_error(gamma=1.2e-5, t=T)

        # filter stage table to single frequencies
        stage_table = stage_table[stage_table['category'].str.contains('1freq')]

        # get optimized concurrence table
        (__, t_PE), (__, t_SQ) = stage_table.groupby('target', sort=True)
        C_opt_table = t_PE\
                .groupby(['w1 [GHz]', 'w2 [GHz]', 'wc [GHz]'], as_index=False)\
                .apply(lambda df: df.sort('J_PE').head(1))\
                .reset_index(level=0, drop=True)

        t_PE_weyl = t_PE[(t_PE['max loss']<0.1) & (t_PE['C']==1.0)]
        weyl = QDYN.weyl.WeylChamber()

        # table of zetas at the same data points as C_opt_table
        ind = ['w1 [GHz]', 'w2 [GHz]', 'wc [GHz]']
        zeta_table2 = pd.merge(C_opt_table[ind+['C', 'max loss']],
                              zeta_table[ind+['zeta [MHz]']],
                              on=ind, how='left').dropna()
        zeta = zeta_table2['zeta [MHz]']
        gamma = -2.0 * np.pi * (zeta/1000.0) * T # entangling phase
        C_ff = np.abs(np.sin(0.5*gamma))

        # row 1: Weyl chamber
        pos = [(left_margin+i_col*(w+hgap)+weyl_offset_x)/fig_width,
               (bottom_margin+2*(h+vgap)+weyl_offset_y)/fig_height,
               weyl_width/fig_width, weyl_height/fig_height]
        ax_weyl = fig.add_axes(pos, projection='3d');
        weyl.scatter(t_PE_weyl['c1'], t_PE_weyl['c2'], t_PE_weyl['c3'],
                     s=5, linewidth=0)
        weyl.render(ax_weyl)
        ax_weyl.xaxis._axinfo['ticklabel']['space_factor'] = 1.0
        ax_weyl.yaxis._axinfo['ticklabel']['space_factor'] = 1.0
        ax_weyl.zaxis._axinfo['ticklabel']['space_factor'] = 1.3
        ax_weyl.xaxis._axinfo['label']['space_factor'] = 1.5
        ax_weyl.yaxis._axinfo['label']['space_factor'] = 1.5
        ax_weyl.zaxis._axinfo['label']['space_factor'] = 1.5
        ax_weyl.xaxis.set_major_formatter(weyl_x_tick_fmt)
        ax_weyl.yaxis.set_major_formatter(weyl_y_tick_fmt)
        ax_weyl.zaxis.set_major_formatter(weyl_z_tick_fmt)

        # row 2: 1-C_SQ
        pos = [(left_margin+i_col*(w+hgap))/fig_width,
               (bottom_margin+h+vgap)/fig_height,
               w/fig_width, h/fig_height]
        ax = fig.add_axes(pos);
        if T == 10:
            pos_cbar = [(left_margin+i_col*(w+hgap)+w+cbar_gap)/fig_width,
                        (bottom_margin+h+vgap)/fig_height,
                        cbar_width/fig_width, h/fig_height]
            ax_cbar = fig.add_axes(pos_cbar)
        else:
            ax_cbar = None
        cbar = render_values(C_opt_table['wc [GHz]'], C_opt_table['w2 [GHz]'],
                             C_opt_table['C'], ax, ax_cbar, density=density,
                             vmin=0.0, vmax=1.0, bg='black',
                             val_alpha=(1-C_opt_table['max loss']),
                             transform_x=DeltaC, transform_y=Delta2)
        if ax_cbar is not None:
            ax_cbar.set_ylabel(r'$C_{\text{PE}}$ (opt)', rotation=90)
            cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax_cbar.yaxis.set_ticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
        set_axis(ax, 'y', y_tick0, y_tick1, y_major_ticks,
                range=y_range, minor=y_minor)
        set_axis(ax, 'x', x_tick0, x_tick1, x_major_ticks,
                range=x_range, minor=x_minor, ticklabels=False)
        ax.tick_params(which='both', direction='out')
        if i_col > 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(r"$\Delta_2/\alpha$", labelpad=ylabelpad)
        labels = [
        #          w_c   w_2     label pos
            ("A", (5.75, 6.32 ), (5.35, 6.40), 'FireBrick'),
            ("B", (6.20, 5.90 ), (6.35, 5.95), 'FireBrick')
        ]
        for (label, x_y_data, x_y_label, color) in labels:
            ax.scatter((DeltaC(x_y_data[0]),), (Delta2(x_y_data[1]), ),
                    color=color, marker='x')
            ax.annotate(label, (DeltaC(x_y_label[0]), Delta2(x_y_label[1])),
                        color=color)

        # row 3: C_0-C_SQ
        pos = [(left_margin+i_col*(w+hgap))/fig_width,
               bottom_margin/fig_height,
               w/fig_width, h/fig_height]
        ax = fig.add_axes(pos);
        if T == 10:
            pos_cbar = [(left_margin+i_col*(w+hgap)+w+cbar_gap)/fig_width,
                        bottom_margin/fig_height,
                        cbar_width/fig_width, h/fig_height]
            ax_cbar = fig.add_axes(pos_cbar)
        else:
            ax_cbar = None
        cbar = render_values(zeta_table2['wc [GHz]'], zeta_table2['w2 [GHz]'],
                             zeta_table2['C']-C_ff,
                             ax, ax_cbar, density=density, vmin=0.0, vmax=1.0,
                             val_alpha=(1-zeta_table2['max loss']), bg='black',
                             transform_x=DeltaC, transform_y=Delta2)
        if ax_cbar is not None:
            ax_cbar.set_ylabel(r'$C_{\text{PE}}-C_{0}$', rotation=90)
            cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax_cbar.yaxis.set_ticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
        set_axis(ax, 'y', y_tick0, y_tick1, y_major_ticks,
                range=y_range, minor=y_minor)
        set_axis(ax, 'x', x_tick0, x_tick1, x_major_ticks,
                range=x_range, minor=x_minor)
        ax.tick_params(which='both', direction='out')
        if i_col > 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(r"$\Delta_2/\alpha$", labelpad=ylabelpad)
        ax.set_xlabel(r"$\Delta_c/g$", labelpad=xlabelpad)
        labels = [
        #          w_c   w_2     label pos
            ("A", (5.75, 6.32 ), (5.35, 6.40), 'OrangeRed'),
            ("B", (6.20, 5.90 ), (6.35, 5.95), 'FireBrick')
        ]
        for (label, x_y_data, x_y_label, color) in labels:
            ax.scatter((DeltaC(x_y_data[0]),), (Delta2(x_y_data[1]), ),
                    color=color, marker='x')
            ax.annotate(label, (DeltaC(x_y_label[0]), Delta2(x_y_label[1])),
                        color=color)

        fig.text((left_margin+i_col*(w+hgap)+0.95*w)/fig_width,
                 (bottom_margin+2*(h+vgap)+h+0.3)/fig_height,
                 r'$T = %d$~ns' % T, verticalalignment='top',
                 horizontalalignment='right', size=10)

    if OUTFOLDER is not None:
        outfile = os.path.join(OUTFOLDER, outfile)

    fig.savefig(outfile)
    print("written %s" % outfile)
    plt.close(fig)


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

    fig_height      = 4.5
    fig_width       = 8.5               # Total canvas (cv) width
    left_margin     = 1.2               # Left cv -> plot area
    right_margin    = 0.05              # plot area -> right cv
    top_margin      = 0.05              # top cv -> plot area
    bottom_margin   = 0.6
    gap = 0.15
    h = 0.5*(fig_height - (bottom_margin + top_margin + gap))
    w = fig_width  - (left_margin + right_margin)
    data = r'''
    #                   minimum error   min err (B)     PE error     err(H1)
    # gate duration [ns]
    5                        3.02e-04      3.62e-04     1.10e-03   1.00e-00
    10                       6.03e-04      7.23e-04     6.31e-04   7.34e-02
    20                       1.21e-03      1.45e-03     1.21e-03   2.20e-02
    50                       3.01e-03      3.61e-03     3.01e-03   4.46e-03
    100                      6.00e-03      7.20e-03     6.01e-03   7.65e-03
    200                      1.20e-02      1.43e-02     1.20e-02   1.47e-02
    '''
    # The above data is taken from the following sources:
    #
    # * eps_avg^{H1,B}: See notebook QSL_Hadamard1.ipynb,
    #   runfolders in ./QSL_H1_prop. For 50ns, run in
    #   ./propagate_universal/rho/H_L/
    # * For eps_avg^{PE}: See notebook Stage3Analysis.ipynb,
    #   runfolders in ./liouville_prop/stage3/
    # * For eps_avg^{0,B}: See notebook UniversalPropLiouville.ipynb
    #   runfolders in ./propagate_universal/liouville_ff/
    # * For eps_avg^0: See notebook LiouvilleError.ipynb (analytic formula)
    #
    fig = new_figure(fig_width, fig_height, style=STYLE)

    T, eps_0, eps_0B, eps_PE, eps_H1 = np.genfromtxt(StringIO(data), unpack=True)
    pos = [left_margin/fig_width, (bottom_margin+h+gap)/fig_height,
           w/fig_width, h/fig_height]
    ax = fig.add_axes(pos)
    ax.plot(T, eps_0, label=r'$\varepsilon_{\text{avg}}^0$',
            marker='o', color=get_color('grey'))
    ax.plot(T, eps_PE, label=r'$\varepsilon_{\text{avg}}^{\text{PE}}$',
            color=get_color('orange'), marker='o', dashes=ls['dashed'])
    ax.legend(loc='lower right')
    ax.annotate(r'$\text{QSL\,}_{\text{PE}}$',
                xy=(0.230, 0.3), xycoords='axes fraction',
                xytext=(0.230, 0.7), textcoords='axes fraction',
                arrowprops=dict(facecolor='black', width=0.75, headwidth=2,
                                shrink=0.20),
                horizontalalignment='center', verticalalignment='top',
                )
    set_axis(ax, 'x', 4, 210, label='', logscale=True, ticklabels=False)
    set_axis(ax, 'y', 2e-4, 3.0e-2, label='gate error', logscale=True)
    ax.tick_params(axis='x', pad=3)

    pos = [left_margin/fig_width, bottom_margin/fig_height,
           w/fig_width, h/fig_height]
    ax = fig.add_axes(pos)
    ax.plot(T, eps_0, label=None, marker='o', color=get_color('grey'))
    ax.plot(T, eps_0B, label=r'$\varepsilon_{\text{avg}}^{0,\text{B}}$',
            marker='o', color=get_color('blue'), dashes=ls['dashed'])
    ax.plot(T, eps_H1, label=r'$\varepsilon_{\text{avg}}^{\text{H1,B}}$',
            marker='o', color=get_color('red'), dashes=ls['long-dashed'])
    ax.legend(loc='upper left')
    ax.annotate(r'$\text{QSL\,}^{\text{B}}_{\text{H1}}$',
                xy=(0.637, 0.5), xycoords='axes fraction',
                xytext=(0.7, 0.1), textcoords='axes fraction',
                arrowprops=dict(facecolor='black', width=0.7, headwidth=2,
                                shrink=0.25),
                horizontalalignment='left', verticalalignment='bottom',
                )
    set_axis(ax, 'x', 4, 210, label='gate time (ns)', logscale=True, labelpad=-2)
    set_axis(ax, 'y', 2e-4, 3.0e-2, label='gate error', logscale=True)
    ax.tick_params(axis='x', pad=3)

    if OUTFOLDER is not None:
        outfile = os.path.join(OUTFOLDER, outfile)
    fig.savefig(outfile, format=os.path.splitext(outfile)[1][1:])
    print("written %s" % outfile)


def generate_universal_pulse_plot(universal_rf, field_free_rf, outfile):
    fig_width    = 18.0
    fig_height   = 14.2
    spec_offset  =  0.7
    phase_deriv_offset =  2.95
    phase_offset =  4.45
    pulse_offset = 5.95
    phase_h       =  1.5
    phase_deriv_h =  1.5
    label_offset = 14.0
    error_offset = 13.6
    spec_h       =  1.5
    pulse_h      =  1.5
    left_margin  =  1.4
    right_margin =  0.25
    gap          =  0.0 # horizontal gap between panels
    y_label_offset  = 0.07
    log_offset = 8.3
    log_h = 1.0
    dyn_offset = 10.0
    dyn_width = 1.55

    fig = new_figure(fig_width, fig_height, style=STYLE)

    w = float(fig_width - (left_margin + right_margin + 4 * gap)) / 5

    labels = {
            'H_L': r'Hadamard (1)',
            'H_R': r'Hadamard (2)',
            'S_L': r'Phasegate (1)',
            'S_R': r'Phasegate (2)',
            'PE': r'BGATE',
    }

    errors = { # errors obtained from *Liouville space* propagation, see
               # ./propagate_universal/rho folder
            'H_L': 4.46e-3,
            'H_R': 5.03e-3,
            'S_L': 4.79e-3,
            'S_R': 4.07e-3,
            'PE':  4.70e-3,
    }

    polar_axes = []

    for i_tgt, tgt in enumerate(['H_L', 'H_R', 'S_L', 'S_R', 'PE']):

        left_offset = left_margin + i_tgt * (w+gap)

        p = QDYN.pulse.Pulse(os.path.join(universal_rf[tgt], 'pulse.dat'),
                             freq_unit='MHz')
        #U = QDYN.gate2q.Gate2Q(os.path.join(universal_rf[tgt], 'U.dat'))
        #O = read_target_gate(os.path.join(universal_rf[tgt], 'target_gate.dat'))
        #err = 1-U.F_avg(O) # (non-Hermitian) Hilbert space error
        err = errors[tgt] #  Liouville space error
        freq, spectrum = p.spectrum(mode='abs', sort=True)
        spectrum *= 1.0 / len(spectrum)

        # column labels
        fig.text((left_offset + 0.5*w)/fig_width, label_offset/fig_height,
                  labels[tgt], verticalalignment='top',
                  horizontalalignment='center')

        fig.text((left_offset + 0.5*w)/fig_width, error_offset/fig_height,
                  r'$\varepsilon_{\text{avg}} = %s$' % latex_exp(err),
                  verticalalignment='top', horizontalalignment='center')

        # spectrum
        pos = [left_offset/fig_width, spec_offset/fig_height,
               w/fig_width, spec_h/fig_height]
        ax_spec = fig.add_axes(pos)
        ax_spec.plot(freq, 1.1*spectrum, label='spectrum')
        set_axis(ax_spec, 'x', -1000, 1000, range=(-650, 600), step=500, minor=5,
                 label=r'$\Delta f$ (MHz)', labelpad=1)
        w1 = 5.9823 # GHz
        w2 = 5.8824 # GHz
        wd = 5.9325 # GHz
        ax_spec.axvline(x=1000*(w2-wd), ls='--', color=get_color('green'))
        ax_spec.axvline(x=1000*(w1-wd), ls='--', color=get_color('orange'))
        ax_spec.text(x=1000*(w2-wd)-50, y=90, s=r'$\omega_2^d$',
                     ha='right', va='top', color=get_color('green'))
        ax_spec.text(x=1000*(w1-wd)+50, y=90, s=r'$\omega_1^d$',
                     ha='left', va='top', color=get_color('orange'))
        if i_tgt == 0:
            set_axis(ax_spec, 'y', 0, 100, step=50, minor=2, label='')
        else:
            set_axis(ax_spec, 'y', 0, 100, step=50, minor=2, label='',
                     ticklabels=False)

        # phase
        pos = [left_offset/fig_width, phase_deriv_offset/fig_height,
               w/fig_width, phase_deriv_h/fig_height]
        ax_phase_deriv = fig.add_axes(pos)
        ax_phase_deriv.plot(p.tgrid, p.phase(unwrap=True, s=1000,
                            derivative=True))
        if i_tgt < 4:
            set_axis(ax_phase_deriv, 'x', 0, 50, step=10, minor=2,
                     label='time (ns)', labelpad=1, drop_ticklabels=[-1, ])
        else:
            set_axis(ax_phase_deriv, 'x', 0, 50, step=10, minor=2,
                     label='time (ns)', labelpad=1)
        if i_tgt == 0:
            set_axis(ax_phase_deriv, 'y', -500, 500, range=(-400, 250),
                     step=200, minor=2, label='')
        else:
            set_axis(ax_phase_deriv, 'y', -500, 500, range=(-400, 250),
                     step=200, minor=2, label='', ticklabels=False)
        ax_phase_deriv.axhline(y=1000*(w2-wd), ls='--',
                               color=get_color('green'))
        ax_phase_deriv.axhline(y=1000*(w1-wd), ls='--',
                               color=get_color('orange'))

        pos = [left_offset/fig_width, phase_offset/fig_height,
               w/fig_width, phase_h/fig_height]
        ax_phase = fig.add_axes(pos)
        ax_phase.plot(p.tgrid, p.phase(unwrap=True) / np.pi)
        set_axis(ax_phase, 'x', 0, 50, step=10, minor=2, label='',
                 ticklabels=False, labelpad=1)
        if i_tgt == 0:
            set_axis(ax_phase, 'y', -16, 4, range=(-15, 5), step=4, minor=2,
                    label='', drop_ticklabels=[-1, ])
        else:
            set_axis(ax_phase, 'y', -16, 16, range=(-14.9, 4.9), step=4,
                     minor=2, label='', ticklabels=False)

        # pulse
        pos = [left_offset/fig_width, pulse_offset/fig_height,
               w/fig_width, pulse_h/fig_height]
        ax_pulse = fig.add_axes(pos)
        p.render_pulse(ax_pulse)
        avg_pulse = np.trapz(np.abs(p.amplitude), p.tgrid) / p.tgrid[-1]
        ax_pulse.axhline(y=avg_pulse, color='black', dashes=ls['dotted'])
        set_axis(ax_pulse, 'x', 0, 50, step=10, minor=2, #label='time (ns)',
                 label='', ticklabels=False,
                 labelpad=1)
        if i_tgt == 0:
            set_axis(ax_pulse, 'y', 0, 300, step=100, minor=2, label='')
        else:
            set_axis(ax_pulse, 'y', 0, 300, step=100, minor=2, label='',
                     ticklabels=False)

        # logical subspace population
        pos = [left_offset/fig_width,log_offset/fig_height,
               w/fig_width, log_h/fig_height]
        ax_log = fig.add_axes(pos)
        dyn = QDYNTransmonLib.popdyn.PopPlot(universal_rf[tgt])
        pop_loss = np.zeros(len(dyn.tgrid))
        for i_state, basis_state in enumerate(['11', '10', '01', '00']):
            pop_loss += 0.25*(  dyn.pop[basis_state].pop00
                              + dyn.pop[basis_state].pop01
                              + dyn.pop[basis_state].pop10
                              + dyn.pop[basis_state].pop11)
        pop_loss = 1.0 - pop_loss
        avg_loss = np.trapz(pop_loss, dyn.tgrid) / dyn.tgrid[-1]
        ax_log.fill(dyn.tgrid, pop_loss, color=get_color('grey'))
        ax_log.axhline(y=avg_loss, color='black', dashes=ls['dotted'])
        if i_tgt < 4:
            set_axis(ax_log, 'x', 0, 50, step=10, minor=2, label='time (ns)',
                    labelpad=1, drop_ticklabels=[-1, ])
        else:
            set_axis(ax_log, 'x', 0, 50, step=10, minor=2, label='time (ns)',
                    labelpad=1)
        if i_tgt == 0:
            set_axis(ax_log, 'y', 0, 0.3, range=(0,0.25), step=0.1, minor=2,
                     label='')
        else:
            set_axis(ax_log, 'y', 0, 0.3, range=(0,0.25), step=0.1, minor=2,
                     label='', ticklabels=False)

        # population dynamics
        tgrid, psi01_ff_re, psi01_ff_im = np.genfromtxt(
                        join(field_free_rf, 'psi01_phases.dat'),
                        usecols=(0,3,4), unpack=True)
        phase01_ff = np.unwrap(np.arctan2(psi01_ff_re, psi01_ff_im))
        E01 = (phase01_ff/tgrid)[-1]
        psi10_ff_re, psi10_ff_im = np.genfromtxt(
                        join(field_free_rf, 'psi10_phases.dat'), usecols=(5,6),
                        unpack=True)
        phase10_ff = np.unwrap(np.arctan2(psi10_ff_re, psi10_ff_im))
        E10 = (phase10_ff/tgrid)[-1]
        psi11_ff_re, psi11_ff_im = np.genfromtxt(
                        join(field_free_rf, 'psi11_phases.dat'), usecols=(7,8),
                        unpack=True)
        phase11_ff = np.unwrap(np.arctan2(psi11_ff_re, psi11_ff_im))
        E11 = (phase11_ff/tgrid)[-1]

        tgrid, psi00_00_re, psi00_00_im, psi00_01_re, psi00_01_im, \
        psi00_10_re, psi00_10_im, psi00_11_re, psi00_11_im \
        = np.genfromtxt(join(universal_rf[tgt], 'psi00_phases.dat'),
                        unpack=True)
        phase00_00 = np.unwrap(np.arctan2(psi00_00_re, psi00_00_im))
        phase00_01 = np.unwrap(np.arctan2(psi00_01_re, psi00_01_im))-E01*tgrid
        phase00_10 = np.unwrap(np.arctan2(psi00_10_re, psi00_10_im))-E10*tgrid
        phase00_11 = np.unwrap(np.arctan2(psi00_11_re, psi00_11_im))-E11*tgrid
        r00_00 = np.sqrt(psi00_00_re**2 + psi00_00_im**2)
        r00_01 = np.sqrt(psi00_01_re**2 + psi00_01_im**2)
        r00_10 = np.sqrt(psi00_10_re**2 + psi00_10_im**2)
        r00_11 = np.sqrt(psi00_11_re**2 + psi00_11_im**2)

        tgrid, psi01_00_re, psi01_00_im, psi01_01_re, psi01_01_im, \
        psi01_10_re, psi01_10_im, psi01_11_re, psi01_11_im \
        = np.genfromtxt(join(universal_rf[tgt], 'psi01_phases.dat'),
                        unpack=True)
        phase01_00 = np.unwrap(np.arctan2(psi01_00_re, psi01_00_im))
        phase01_01 = np.unwrap(np.arctan2(psi01_01_re, psi01_01_im))-E01*tgrid
        phase01_10 = np.unwrap(np.arctan2(psi01_10_re, psi01_10_im))-E10*tgrid
        phase01_11 = np.unwrap(np.arctan2(psi01_11_re, psi01_11_im))-E11*tgrid
        r01_00 = np.sqrt(psi01_00_re**2 + psi01_00_im**2)
        r01_01 = np.sqrt(psi01_01_re**2 + psi01_01_im**2)
        r01_10 = np.sqrt(psi01_10_re**2 + psi01_10_im**2)
        r01_11 = np.sqrt(psi01_11_re**2 + psi01_11_im**2)

        tgrid, psi10_00_re, psi10_00_im, psi10_01_re, psi10_01_im, \
        psi10_10_re, psi10_10_im, psi10_11_re, psi10_11_im \
        = np.genfromtxt(join(universal_rf[tgt], 'psi10_phases.dat'),
                        unpack=True)
        phase10_00 = np.unwrap(np.arctan2(psi10_00_re, psi10_00_im))
        phase10_01 = np.unwrap(np.arctan2(psi10_01_re, psi10_01_im))-E01*tgrid
        phase10_10 = np.unwrap(np.arctan2(psi10_10_re, psi10_10_im))-E10*tgrid
        phase10_11 = np.unwrap(np.arctan2(psi10_11_re, psi10_11_im))-E11*tgrid
        r10_00 = np.sqrt(psi10_00_re**2 + psi10_00_im**2)
        r10_01 = np.sqrt(psi10_01_re**2 + psi10_01_im**2)
        r10_10 = np.sqrt(psi10_10_re**2 + psi10_10_im**2)
        r10_11 = np.sqrt(psi10_11_re**2 + psi10_11_im**2)

        tgrid, psi11_00_re, psi11_00_im, psi11_01_re, psi11_01_im, \
        psi11_10_re, psi11_10_im, psi11_11_re, psi11_11_im \
        = np.genfromtxt(join(universal_rf[tgt], 'psi11_phases.dat'),
                        unpack=True)
        phase11_00 = np.unwrap(np.arctan2(psi11_00_re, psi11_00_im))
        phase11_01 = np.unwrap(np.arctan2(psi11_01_re, psi11_01_im))-E01*tgrid
        phase11_10 = np.unwrap(np.arctan2(psi11_10_re, psi11_10_im))-E10*tgrid
        phase11_11 = np.unwrap(np.arctan2(psi11_11_re, psi11_11_im))-E11*tgrid
        r11_00 = np.sqrt(psi11_00_re**2 + psi11_00_im**2)
        r11_01 = np.sqrt(psi11_01_re**2 + psi11_01_im**2)
        r11_10 = np.sqrt(psi11_10_re**2 + psi11_10_im**2)
        r11_11 = np.sqrt(psi11_11_re**2 + psi11_11_im**2)

        dyn_h_offset = 0.5*(w - 2*dyn_width)
        pos00 = [(left_offset+dyn_h_offset)/fig_width,
                 (dyn_offset+dyn_width)/fig_height,
                 dyn_width/fig_width, dyn_width/fig_height]
        pos01 = [(left_offset+dyn_h_offset+dyn_width)/fig_width,
                 (dyn_offset+dyn_width)/fig_height,
                 dyn_width/fig_width, dyn_width/fig_height]
        pos10 = [(left_offset+dyn_h_offset)/fig_width,
                 dyn_offset/fig_height,
                 dyn_width/fig_width, dyn_width/fig_height]
        pos11 = [(left_offset+dyn_h_offset+dyn_width)/fig_width,
                 dyn_offset/fig_height,
                 dyn_width/fig_width, dyn_width/fig_height]
        ax00 = fig.add_axes(pos00, projection='polar')
        ax01 = fig.add_axes(pos01, projection='polar')
        ax10 = fig.add_axes(pos10, projection='polar')
        ax11 = fig.add_axes(pos11, projection='polar')
        polar_axes.extend([ax00, ax01, ax10, ax11])
        if i_tgt == 0:
            fig.text((left_offset + dyn_h_offset-0.1)/fig_width,
                    (dyn_offset+dyn_width)/fig_height, rotation='vertical',
                    s=r'$\Im[\Psi(t)]$', verticalalignment='center',
                    horizontalalignment='right')
            fig.text((left_offset + dyn_h_offset-0.6)/fig_width,
                    (dyn_offset+0.25*dyn_width)/fig_height, rotation='vertical',
                    s=r'$\ket{00}$', verticalalignment='center',
                    horizontalalignment='right', color=get_color('blue'))
            fig.text((left_offset + dyn_h_offset-0.6)/fig_width,
                    (dyn_offset+0.75*dyn_width)/fig_height, rotation='vertical',
                    s=r'$\ket{01}$', verticalalignment='center',
                    horizontalalignment='right', color=get_color('orange'))
            fig.text((left_offset + dyn_h_offset-0.6)/fig_width,
                    (dyn_offset+1.25*dyn_width)/fig_height, rotation='vertical',
                    s=r'$\ket{10}$', verticalalignment='center',
                    horizontalalignment='right', color=get_color('red'))
            fig.text((left_offset + dyn_h_offset-0.6)/fig_width,
                    (dyn_offset+1.75*dyn_width)/fig_height, rotation='vertical',
                    s=r'$\ket{11}$', verticalalignment='center',
                    horizontalalignment='right', color=get_color('green'))
        fig.text((left_offset + dyn_h_offset + dyn_width)/fig_width,
                (dyn_offset-0.1)/fig_height,
                s=r'$\Re[\Psi(t)]$', verticalalignment='top',
                horizontalalignment='center')

        ax00.plot(phase00_00, r00_00, color=get_color('blue'),   lw=0.7)
        ax00.plot(phase00_01, r00_01, color=get_color('orange'), lw=0.7)
        ax00.plot(phase00_10, r00_10, color=get_color('red'),    lw=0.7)
        ax00.plot(phase00_11, r00_10, color=get_color('green'),  lw=0.7)
        ax01.plot(phase01_00, r01_00, color=get_color('blue'),   lw=0.7)
        ax01.plot(phase01_01, r01_01, color=get_color('orange'), lw=0.7)
        ax01.plot(phase01_10, r01_10, color=get_color('red'),    lw=0.7)
        ax01.plot(phase01_11, r01_11, color=get_color('green'),  lw=0.7)
        ax10.plot(phase10_00, r10_00, color=get_color('blue'),   lw=0.7)
        ax10.plot(phase10_01, r10_01, color=get_color('orange'), lw=0.7)
        ax10.plot(phase10_10, r10_10, color=get_color('red'),    lw=0.7)
        ax10.plot(phase10_11, r10_11, color=get_color('green'),  lw=0.7)
        ax11.plot(phase11_00, r11_00, color=get_color('blue'),   lw=0.7)
        ax11.plot(phase11_01, r11_01, color=get_color('orange'), lw=0.7)
        ax11.plot(phase11_10, r11_10, color=get_color('red'),    lw=0.7)
        ax11.plot(phase11_11, r11_11, color=get_color('green'),  lw=0.7)
        ax00.scatter((phase00_00[0], ), (r00_00[0], ), c=(get_color('blue'),),
                     marker='s')
        ax00.scatter(
            (phase00_00[-1], phase00_01[-1], phase00_10[-1], phase00_11[-1]),
            (r00_00[-1], r00_01[-1], r00_10[-1], r00_11[-1]),
            c = [get_color(clr) for clr in ['blue', 'orange', 'red', 'green']],
            lw=0.5
        )
        ax01.scatter((phase01_01[0], ), (r01_01[0], ), c=(get_color('orange'),),
                     marker='s')
        ax01.scatter(
            (phase01_00[-1], phase01_01[-1], phase01_10[-1], phase01_11[-1]),
            (r01_00[-1], r01_01[-1], r01_10[-1], r01_11[-1]),
            c = [get_color(clr) for clr in ['blue', 'orange', 'red', 'green']],
            lw=0.5
        )
        ax10.scatter((phase10_10[0], ), (r10_10[0], ), c=(get_color('red'),),
                     marker='s')
        ax10.scatter(
            (phase10_00[-1], phase10_01[-1], phase10_10[-1], phase10_11[-1]),
            (r10_00[-1], r10_01[-1], r10_10[-1], r10_11[-1]),
            c = [get_color(clr) for clr in ['blue', 'orange', 'red', 'green']],
            lw=0.5
        )
        ax11.scatter((phase11_11[0], ), (r11_11[0], ), c=(get_color('green'),),
                     marker='s')
        ax11.scatter(
            (phase11_00[-1], phase11_01[-1], phase11_10[-1], phase11_11[-1]),
            (r11_00[-1], r11_01[-1], r11_10[-1], r11_11[-1]),
            c = [get_color(clr) for clr in ['blue', 'orange', 'red', 'green']],
            lw=0.5
        )

    for ax in polar_axes:
        ax.grid(False)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.plot(np.linspace(0, 2*np.pi, 50), np.ones(50)/np.sqrt(2.0), lw=0.5,
                dashes=ls['dashed'], color='black')
        ax.set_rmax(1.0)

    fig.text(y_label_offset/fig_width,
                (spec_offset+0.5*spec_h)/fig_height,
                r'$\vert F(\epsilon) \vert$ (arb. un.)',
                rotation='vertical', va='center', ha='left')
    fig.text(y_label_offset/fig_width,
                (phase_offset+0.5*phase_h)/fig_height,
                r'$\phi$ ($\pi$)',
                rotation='vertical', va='center', ha='left')
    fig.text(y_label_offset/fig_width,
                (phase_deriv_offset+0.5*phase_deriv_h)/fig_height,
                r'$\frac{d\phi}{dt}$ (MHz)',
                rotation='vertical', va='center', ha='left')
    fig.text(y_label_offset/fig_width,
                (pulse_offset+0.5*pulse_h)/fig_height,
                r'$\vert\epsilon\vert$ (MHz)',
                rotation='vertical', va='center', ha='left')
    fig.text(y_label_offset/fig_width,
                (log_offset+0.5*log_h)/fig_height,
                r'$P_\text{outside}$',
                rotation='vertical', va='center', ha='left')

    if OUTFOLDER is not None:
        outfile = os.path.join(OUTFOLDER, outfile)
    fig.savefig(outfile, format=os.path.splitext(outfile)[1][1:])
    print("written %s" % outfile)


def latex_exp(f):
    """Convert float to scientific notation in LaTeX"""
    str = "%.2e" % f
    mantissa, exponent = str.split("e")
    return r'%.2f \times 10^{%d}' % (float(mantissa), int(exponent))


def main(argv=None):

    if argv is None:
        argv = sys.argv
    if not os.path.isdir(OUTFOLDER):
        QDYN.shutil.mkdir(OUTFOLDER)

    stage_table_200 = get_stage3_table('./runs_200_RWA')
    stage_table_050 = get_stage3_table('./runs_020_RWA')
    stage_table_010 = get_stage3_table('./runs_010_RWA')
    zeta_table = get_zeta_table('./runs_050_RWA', T=50)


    universal_root = './runs_zeta_detailed/w2_5900MHz_wc_6200MHz'
    universal_rf = {
        'H_L': universal_root+'/50ns_w_center_H_left',
        'H_R': universal_root+'/50ns_w_center_H_right',
        'S_L': universal_root+'/50ns_w_center_Ph_left',
        'S_R': universal_root+'/50ns_w_center_Ph_right',
        'PE':  universal_root+'/PE_LI_BGATE_50ns_cont_SM'
    }
    field_free_rf = universal_root+'/analyze_ham'

    # Fig 1
    generate_field_free_plot(zeta_table, T=50, outfile='fig1_main.pdf')

    # Fig 2
    generate_map_plot_SQ(stage_table_200, stage_table_050, stage_table_010,
                      zeta_table, outfile='fig2_main.pdf')

    # Fig 3
    generate_map_plot_PE(stage_table_200, stage_table_050, stage_table_010,
                         zeta_table, outfile='fig3_main.pdf')
    # Fig 4
    generate_error_plot(outfile='fig4_main.pdf')

    # Fig 5
    generate_universal_pulse_plot(universal_rf, field_free_rf,
                                  outfile='fig5.pdf')

if __name__ == "__main__":
    sys.exit(main())
