#!/usr/bin/env python
import os
from os.path import join
import sys
from StringIO import StringIO
import QDYN
import QDYNTransmonLib
import numpy as np
from numpy import sin, cos, pi
import matplotlib
from collections import OrderedDict
matplotlib.use('Agg')
import matplotlib.pylab as plt
from notebook_utils import diss_error, render_values
from notebook_utils import get_stage3_table, get_zeta_table
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


def generate_field_free_plot(zeta_table, T, outfile, cols=(1,2,3)):
    """Plot field-free entangling energy zeta, and projected concurrence after
    the given gate duration T in ns.
    """
    left_margin   = 1.05
    cbar_width    = 0.25
    cbar_gap      = 0.25
    hgap1         = 1.13
    hgap2         = 0.93
    right_margin  = 0.75
    w             = 4.2

    # the buffer space we need on the right of the color bar for panel 1, 2, 3
    hgap = [hgap1, hgap2, right_margin]

    top_margin    = 0.65
    bottom_margin = 0.85
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

    def polar(r, phi):
        dx = r * cos(2*pi*phi/360.0)
        dy = r * sin(2*pi*phi/360.0)
        return dx, dy

    n_cols = len(cols)
    fig_height = bottom_margin + top_margin + h
    fig_width = (left_margin + n_cols * (cbar_gap + cbar_width + w) +
                 sum([hgap[col-1] for col in cols]))

    fig = new_figure(fig_width, fig_height, style=STYLE)
    axs = []
    cbar_axs = []

    # Zeta
    zeta = zeta_table['zeta [MHz]']
    abs_zeta = np.clip(np.abs(zeta), a_min=1e-5, a_max=1e5)
    w2 = zeta_table['w2 [GHz]']
    wc = zeta_table['wc [GHz]']

    if 1 in cols:
        i_col = cols.index(1)  # *actual* column number

        pos_x = (
            left_margin + i_col * (cbar_gap + cbar_width + w) +
            sum([hgap[col-1] for col in cols[0:i_col]]))
        pos = [
            pos_x/fig_width, bottom_margin/fig_height,
            w/fig_width, h/fig_height]
        ax = fig.add_axes(pos); axs.append(ax)
        pos_cbar = [
            (pos_x + w + cbar_gap) / fig_width, bottom_margin/fig_height,
            cbar_width/fig_width, h/fig_height]
        ax_cbar = fig.add_axes(pos_cbar)
        cbar_axs.append(ax_cbar)
        cbar = render_values(
            wc, w2, abs_zeta, ax, ax_cbar, density=density,
            logscale=True, vmin=1e-1, transform_x=DeltaC, transform_y=Delta2)
        cbar.ax.yaxis.set_ticks(
            cbar.norm(
                np.concatenate([
                    np.arange(0.1, 1, 0.1), np.arange(1, 10, 1),
                    np.arange(10, 100, 10)])),
            minor=True)
        set_axis(
            ax, 'y', y_tick0, y_tick1, y_major_ticks, range=y_range,
            minor=y_minor, ticklabels=(i_col==0))
        set_axis(
            ax, 'x', x_tick0, x_tick1, x_major_ticks, range=x_range,
            minor=x_minor)
        ax.tick_params(which='both', direction='out')
        if i_col == 0:
            ax.set_ylabel(r"$\Delta_2/\alpha$", labelpad=ylabelpad)
        ax.set_xlabel(r"$\Delta_c/g$", labelpad=xlabelpad)
        fig.text(
            (pos_x + w + cbar_gap + cbar_width+0.53)/fig_width,
            1-0.2/fig_height, r'$\zeta$~(MHz)', verticalalignment='top',
            horizontalalignment='right')
        labels = [
        #         w_c   w_2     label pos
            ("", (6.20, 5.90 ), (6.35, 5.95), 'OrangeRed')
        ]
        for (label, x_y_data, x_y_label, color) in labels:
            ax.scatter(
                (DeltaC(x_y_data[0]),), (Delta2(x_y_data[1]), ),
                color=color, marker='x')
            ax.annotate(label, (DeltaC(x_y_label[0]), Delta2(x_y_label[1])),
                        color=color)

    # Entanglement time

    if 2 in cols:

        i_col = cols.index(2)  # *actual* column number

        T_entangling = 500.0/abs_zeta

        pos_x = (
            left_margin + i_col * (cbar_gap + cbar_width + w) +
            sum([hgap[col-1] for col in cols[0:i_col]]))
        pos = [
            pos_x/fig_width, bottom_margin/fig_height,
            w/fig_width, h/fig_height]
        ax = fig.add_axes(pos)
        pos_cbar = [(pos_x + w + cbar_gap)/fig_width,
                    bottom_margin/fig_height,
                    cbar_width/fig_width, h/fig_height]
        ax_cbar = fig.add_axes(pos_cbar)
        cbar = render_values(
            wc, w2, T_entangling, ax, ax_cbar, density=density, logscale=True,
            vmax=1e3, transform_x=DeltaC, transform_y=Delta2)
        cbar.ax.yaxis.set_ticks(cbar.norm(np.concatenate(
                [np.arange(4, 10, 1), np.arange(10, 100, 10),
                np.arange(100, 1000, 100)])), minor=True)
        set_axis(
            ax, 'y', y_tick0, y_tick1, y_major_ticks, range=y_range,
            minor=y_minor, ticklabels=(i_col==0))
        set_axis(
            ax, 'x', x_tick0, x_tick1, x_major_ticks, range=x_range,
            minor=x_minor)
        ax.tick_params(which='both', direction='out')
        if i_col == 0:
            ax.set_ylabel(r"$\Delta_2/\alpha$", labelpad=ylabelpad)
        ax.set_xlabel(r"$\Delta_c/g$", labelpad=xlabelpad)
        fig.text(
            (pos_x + w + cbar_gap + cbar_width+0.53)/fig_width,
            1-0.2/fig_height, r'$T(C_0=1)$ (ns)', verticalalignment='top',
            horizontalalignment='right')
        labels = [
        #          w_c   w_2     label pos
            ("", (6.20, 5.90 ), (6.35, 5.95), 'FireBrick')
        ]
        for (label, x_y_data, x_y_label, color) in labels:
            ax.scatter((DeltaC(x_y_data[0]),), (Delta2(x_y_data[1]), ),
                    color=color, marker='x')
            ax.annotate(label, (DeltaC(x_y_label[0]), Delta2(x_y_label[1])),
                        color=color)
        other_gates = [
            # label Dc    D2            r    phi (label)   #                  T [ns]
            ("1",  'o', (15,   1),     (5,  45), 'Black'), #                  150  \cite{LeekPRB2009}
            ("2",  '>', (42,   0.85),  (5, 180), 'Black'), # (Delta_c = 58)   220  \cite{ChowPRL2011}
            ("3",  'o', (26,   0.29),  (6, -50), 'Black'), #                  110  \cite{ChowPRL2012}
            ("4",  '>', (42,   1.1 ),  (6, 115), 'Black'), # (Delta_c = 95)   200  \cite{PolettoPRL2012}
            ("5",  'o', (24,   2.3 ),  (5,  45), 'Black'), #                  500  \cite{ChowNJP2013}
            ("6",  '>', (42,   0.6 ),  (6,-115), 'Black'), # (Delta_c = 43)   350  \cite{ChowNC2014}
            ("7",  'o', (29,   0.6 ),  (5,   0), 'Black'), #                  350  \cite{CorcolesNC2015}
            ("8",  'o', ( 1.4, 1.76),  (5,  45), 'Black'), #                   50  \cite{EconomouPRB2015}
            ("9",  'o', (17,   0.17),  (6, -50), 'Black'), #                  120  \cite{CrossPRA2015}
            ("10", 'o', (25,   0.6 ),  (6, 180), 'Black'), #                  200  \cite{1603.04821}
        ]
        for (label, marker, x_y_data, r_phi_label, color) in other_gates:
            ax.scatter((x_y_data[0], ), (x_y_data[1], ), color=color,
                    marker=marker, s=1.5)
            r, phi = r_phi_label
            ax.annotate(label, xy=x_y_data, xycoords='data', color=color,
                        xytext=polar(r, phi), textcoords='offset points',
                        ha='center', va='center')
                        #xytext=(3,0), textcoords='offset points',
        # guides for Delta_2 = 0, \pm 2 alpha
        # These are broken up into several segments to make them appear "in the
        # background"
        ax.plot([-22, -13], [0, 0], ls='dotted', color='Gray', lw=1) # hline
        ax.plot([10, 15],   [0, 0], ls='dotted', color='Gray', lw=1) # hline
        ax.plot([21, 26],   [0, 0], ls='dotted', color='Gray', lw=1) # hline
        ax.plot([30, 39],   [0, 0], ls='dotted', color='Gray', lw=1) # hline
        ax.plot([-22, -6], [2, 2], ls='dotted', color='Gray', lw=1) # hline
        ax.plot([12, 43], [2, 2], ls='dotted', color='Gray', lw=1) # hline
        ax.plot([-22, -14], [-2, -2], ls='dotted', color='Gray', lw=1) # hline
        ax.plot([4, 43], [-2, -2], ls='dotted', color='Gray', lw=1) # hline

    # Relative effective decay rate

    if 3 in cols:
        i_col = cols.index(3)  # *actual* column number
        gamma_bare = 0.012
        rel_decay = zeta_table['gamma [MHz]'] / gamma_bare
        print("Min: %s" % np.min(rel_decay))
        print("Max: %s" % np.max(rel_decay))

        pos_x = (
            left_margin + i_col * (cbar_gap + cbar_width + w) +
            sum([hgap[col-1] for col in cols[0:i_col]]))
        pos = [
            pos_x/fig_width, bottom_margin/fig_height,
            w/fig_width, h/fig_height]
        ax = fig.add_axes(pos)
        pos_cbar = [
            (pos_x + w + cbar_gap) / fig_width, bottom_margin/fig_height,
            cbar_width/fig_width, h/fig_height]
        ax_cbar = fig.add_axes(pos_cbar)
        cbar = render_values(
            wc, w2, rel_decay, ax, ax_cbar, density=density, logscale=False,
            vmin=1, vmax=2.3, cmap=plt.cm.cubehelix_r, transform_x=DeltaC,
            transform_y=Delta2)
        cbar.set_ticks([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2])
        set_axis(
            ax, 'y', y_tick0, y_tick1, y_major_ticks, range=y_range,
            minor=y_minor, ticklabels=(i_col==0))
        set_axis(
            ax, 'x', x_tick0, x_tick1, x_major_ticks, range=x_range,
            minor=x_minor)
        ax.tick_params(which='both', direction='out')
        if i_col == 0:
            ax.set_ylabel(r"$\Delta_2/\alpha$", labelpad=ylabelpad)
        ax.set_xlabel(r"$\Delta_c/g$", labelpad=xlabelpad)
        fig.text(
            (pos_x + w + cbar_gap + cbar_width +
             right_margin - 0.09)/fig_width,
            1-0.2/fig_height,
            r'$\gamma_{\text{dressed}} / \gamma_{\text{bare}}$',
            verticalalignment='top', horizontalalignment='right')
        labels = [
        #          w_c   w_2     label pos
            ("", (6.20, 5.90 ), (6.35, 5.95), 'FireBrick')
        ]
        for (label, x_y_data, x_y_label, color) in labels:
            ax.scatter(
                (DeltaC(x_y_data[0]),), (Delta2(x_y_data[1]), ),
                color=color, marker='x')
            ax.annotate(
                label, (DeltaC(x_y_label[0]), Delta2(x_y_label[1])),
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


def generate_map_plot_combined(
        stage_table_200, stage_table_050, stage_table_010, zeta_table, outfile,
        rows=(1,2,3,4)):
    """Table of plots showing results for entanglement minimization /
    maximization

    By default, produces figure with 4 rows in the paper. By passing tuple of
    `rows` that leaves out any of the values 1, 2, 3, 4, a plot that contains
    only a subset of the default rows may be generated.
    """

    left_margin   = 1.1
    hgap          = 0.35
    cbar_width    = 0.25
    cbar_gap      = hgap
    right_margin  = 1.0
    w             = 4.2

    top_margin    = 0.25
    bottom_margin = 0.85
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

    n_rows = len(rows)

    Delta2 = lambda w2: (w2 - w1)/alpha
    DeltaC = lambda wc: (wc - w1)/g
    y_range = (Delta2(w2_min), Delta2(w2_max))
    x_range = (DeltaC(wc_min), DeltaC(wc_max))

    fig_height = bottom_margin + n_rows*h + (n_rows-1)*vgap + top_margin
    fig_width  = (left_margin + 3*w + 2*hgap + cbar_gap + cbar_width
                  + right_margin)
    fig = new_figure(fig_width, fig_height, style=STYLE)

    data = OrderedDict([
            (200, stage_table_200),
            (50,  stage_table_050),
            (10,  stage_table_010), ])

    for i_col, T in enumerate(data.keys()):

        zeta = zeta_table['zeta [MHz]']
        gamma = -2.0 * np.pi * (zeta/1000.0) * T # entangling phase
        C_ff = np.abs(np.sin(0.5*gamma))

        # row 1: field-free entanglement
        if 1 in rows:
            i_row = rows.index(1) + 1  # the *actual* row number
            pos = [(left_margin+i_col*(w+hgap))/fig_width,
                   (bottom_margin+(n_rows-i_row)*(h+vgap))/fig_height,
                   w/fig_width, h/fig_height]
            ax = fig.add_axes(pos);
            if T == 10:
                pos_cbar = [(left_margin+i_col*(w+hgap)+w+cbar_gap)/fig_width,
                            (bottom_margin+(n_rows-i_row)*(h+vgap))/fig_height,
                            cbar_width/fig_width, h/fig_height]
                ax_cbar = fig.add_axes(pos_cbar)
            else:
                ax_cbar = None
            cbar = render_values(
                    zeta_table['wc [GHz]'], zeta_table['w2 [GHz]'],
                    C_ff, ax, ax_cbar, density=density, vmin=0.0, vmax=1.0,
                    transform_x=DeltaC, transform_y=Delta2)
            if ax_cbar is not None:
                ax_cbar.set_ylabel(r'$C_0$', rotation=90)
                cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
                ax_cbar.yaxis.set_ticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
            set_axis(ax, 'y', y_tick0, y_tick1, y_major_ticks,
                     range=y_range, minor=y_minor)
            set_axis(ax, 'x', x_tick0, x_tick1, x_major_ticks,
                     range=x_range, minor=x_minor, ticklabels=(rows[-1] == 1))
            ax.tick_params(which='both', direction='out')
            if i_col > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(r"$\Delta_2/\alpha$", labelpad=ylabelpad)
            labels = [
                #      w_2   w_c     label pos
                ("", (6.20, 5.90 ), (6.35, 5.95), 'OrangeRed')
            ]
            for (label, x_y_data, x_y_label, color) in labels:
                ax.scatter(
                    (DeltaC(x_y_data[0]),), (Delta2(x_y_data[1]), ),
                    color=color, marker='x')
                ax.annotate(
                    label, (DeltaC(x_y_label[0]), Delta2(x_y_label[1])),
                    color=color)
            if (i_col == 2):
                fig.text((left_margin+i_col*(w+hgap)+0.95*w)/fig_width,
                        (bottom_margin+(n_rows-i_row)*(h+vgap)+0.2)/fig_height,
                        r'field-free', verticalalignment='bottom',
                        horizontalalignment='right', size=10, color='white')

        # collection OCT data

        stage_table = data[T]

        # filter stage table to single frequencies
        stage_table = stage_table[stage_table['category'].str.contains('1freq')]

        # get optimized concurrence table
        (__, t_PE), (__, t_SQ) = stage_table.groupby('target', sort=True)
        C_opt_table_PE = t_PE\
                .groupby(['w1 [GHz]', 'w2 [GHz]', 'wc [GHz]'], as_index=False)\
                .apply(lambda df: df.sort('J_PE').head(1))\
                .reset_index(level=0, drop=True)
        C_opt_table_SQ = t_SQ\
                .groupby(['w1 [GHz]', 'w2 [GHz]', 'wc [GHz]'], as_index=False)\
                .apply(lambda df: df.sort('J_PE').head(1))\
                .reset_index(level=0, drop=True)

        zeta = zeta_table['zeta [MHz]']
        gamma = -2.0 * np.pi * (zeta/1000.0) * T # entangling phase
        C_ff = np.abs(np.sin(0.5*gamma))

        # table of zetas at the same data points as C_opt_table_PE
        ind = ['w1 [GHz]', 'w2 [GHz]', 'wc [GHz]']
        combined_table = pd.merge(
                            C_opt_table_PE.rename(
                                columns={'C': 'C_PE',
                                         'max loss': 'max loss (PE)'}
                                )[ind+['C_PE', 'max loss (PE)']],
                            C_opt_table_SQ.rename(
                                columns={'C': 'C_SQ',
                                         'max loss': 'max loss (SQ)'}
                                )[ind+['C_SQ', 'max loss (SQ)']],
                            on=ind, how='left').dropna() \
                        .merge(
                            zeta_table[ind+['zeta [MHz]']],
                            on=ind, how='left').dropna()
        zeta = combined_table['zeta [MHz]']
        gamma = -2.0 * np.pi * (zeta/1000.0) * T # entangling phase
        C_ff = np.abs(np.sin(0.5*gamma))

        # row 2: 1-C_SQ
        if 2 in rows:
            i_row = rows.index(2) + 1  # the *actual* row number
            pos = [(left_margin+i_col*(w+hgap))/fig_width,
                   (bottom_margin+(n_rows-i_row)*(h+vgap))/fig_height,
                   w/fig_width, h/fig_height]
            ax = fig.add_axes(pos)
            if T == 10:
                pos_cbar = [(left_margin+i_col*(w+hgap)+w+cbar_gap)/fig_width,
                            (bottom_margin+(n_rows-i_row)*(h+vgap))/fig_height,
                            cbar_width/fig_width, h/fig_height]
                ax_cbar = fig.add_axes(pos_cbar)
            else:
                ax_cbar = None
            vals = np.minimum(np.array(combined_table['C_SQ']), C_ff)
            cbar = render_values(combined_table['wc [GHz]'],
                                 combined_table['w2 [GHz]'],
                                 vals, ax, ax_cbar, density=density,
                                 vmin=0.0, vmax=1.0, bg='black',
                                 val_alpha=(1-combined_table['max loss (SQ)']),
                                 transform_x=DeltaC, transform_y=Delta2)
            if ax_cbar is not None:
                ax_cbar.set_ylabel(r'$C_{\text{SQ}}$', rotation=90)
                cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
                ax_cbar.yaxis.set_ticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
            set_axis(ax, 'y', y_tick0, y_tick1, y_major_ticks,
                     range=y_range, minor=y_minor)
            set_axis(ax, 'x', x_tick0, x_tick1, x_major_ticks,
                     range=x_range, minor=x_minor, ticklabels=(rows[-1] == 2))
            ax.tick_params(which='both', direction='out')
            if i_col > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(r"$\Delta_2/\alpha$", labelpad=ylabelpad)
            labels = [
                #      w_2   w_c     label pos
                ("", (6.20, 5.90 ), (6.35, 5.95), 'OrangeRed')
            ]
            for (label, x_y_data, x_y_label, color) in labels:
                ax.scatter(
                    (DeltaC(x_y_data[0]),), (Delta2(x_y_data[1]), ),
                    color=color, marker='x')
                ax.annotate(
                    label, (DeltaC(x_y_label[0]), Delta2(x_y_label[1])),
                    color=color)
            if (i_col == 2):
                fig.text(
                    (left_margin+i_col*(w+hgap)+0.95*w)/fig_width,
                    (bottom_margin+(n_rows-i_row)*(h+vgap)+0.2)/fig_height,
                    r'minimization', verticalalignment='bottom',
                    horizontalalignment='right', size=10, color='white')

        # row 3: C_PE
        if 3 in rows:
            i_row = rows.index(3) + 1  # the *actual* row number
            pos = [(left_margin+i_col*(w+hgap))/fig_width,
                   (bottom_margin+(n_rows-i_row)*(h+vgap))/fig_height,
                   w/fig_width, h/fig_height]
            ax = fig.add_axes(pos);
            if T == 10:
                pos_cbar = [(left_margin+i_col*(w+hgap)+w+cbar_gap)/fig_width,
                            (bottom_margin+(n_rows-i_row)*(h+vgap))/fig_height,
                            cbar_width/fig_width, h/fig_height]
                ax_cbar = fig.add_axes(pos_cbar)
            else:
                ax_cbar = None
            cbar = render_values(C_opt_table_PE['wc [GHz]'],
                                 C_opt_table_PE['w2 [GHz]'],
                                 C_opt_table_PE['C'], ax, ax_cbar,
                                 density=density,
                                 vmin=0.0, vmax=1.0, bg='black',
                                 val_alpha=(1-C_opt_table_PE['max loss']),
                                 transform_x=DeltaC, transform_y=Delta2)
            if ax_cbar is not None:
                ax_cbar.set_ylabel(r'$C_{\text{PE}}$', rotation=90)
                cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
                ax_cbar.yaxis.set_ticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
            set_axis(ax, 'y', y_tick0, y_tick1, y_major_ticks,
                    range=y_range, minor=y_minor)
            set_axis(ax, 'x', x_tick0, x_tick1, x_major_ticks,
                     range=x_range, minor=x_minor, ticklabels=(rows[-1] == 3))
            ax.tick_params(which='both', direction='out')
            if i_col > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(r"$\Delta_2/\alpha$", labelpad=ylabelpad)
            labels = [
                #      w_c   w_2     label pos
                ("", (6.20, 5.90 ), (6.35, 5.95), 'FireBrick')
            ]

            for (label, x_y_data, x_y_label, color) in labels:
                ax.scatter(
                    (DeltaC(x_y_data[0]),), (Delta2(x_y_data[1]), ),
                    color=color, marker='x')
                ax.annotate(
                    label, (DeltaC(x_y_label[0]), Delta2(x_y_label[1])),
                    color=color)
            if (i_col == 2):
                fig.text(
                    (left_margin+i_col*(w+hgap)+0.95*w)/fig_width,
                    (bottom_margin+(n_rows-i_row)*(h+vgap)+0.2)/fig_height,
                    r'maximization', verticalalignment='bottom',
                    horizontalalignment='right', size=10, color='white')

        # row 4: C_0-C_SQ
        if 4 in rows:
            i_row = rows.index(4) + 1  # the *actual* row number
            pos = [(left_margin+i_col*(w+hgap))/fig_width,
                   (bottom_margin+(n_rows-i_row)*(h+vgap))/fig_height,
                   w/fig_width, h/fig_height]
            ax = fig.add_axes(pos)
            if T == 10:
                pos_cbar = [(left_margin+i_col*(w+hgap)+w+cbar_gap)/fig_width,
                            (bottom_margin+(n_rows-i_row)*(h+vgap))/fig_height,
                            cbar_width/fig_width, h/fig_height]
                ax_cbar = fig.add_axes(pos_cbar)
            else:
                ax_cbar = None
            vals = ((combined_table['C_PE']) *
                    (1 - np.minimum(np.array(combined_table['C_SQ']), C_ff)))
            val_alpha = ((1-combined_table['max loss (PE)']) *
                         (1-combined_table['max loss (SQ)']))
            cbar = render_values(combined_table['wc [GHz]'],
                                 combined_table['w2 [GHz]'], vals,
                                 ax, ax_cbar, density=density,
                                 vmin=0.0, vmax=1.0,
                                 val_alpha=val_alpha, bg='black',
                                 transform_x=DeltaC, transform_y=Delta2)
            if ax_cbar is not None:
                ax_cbar.set_ylabel(r'$C_{\text{PE}} \times (1-C_\text{SQ})$',
                                   rotation=90)
                cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
                ax_cbar.yaxis.set_ticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
            set_axis(ax, 'y', y_tick0, y_tick1, y_major_ticks,
                    range=y_range, minor=y_minor)
            set_axis(ax, 'x', x_tick0, x_tick1, x_major_ticks,
                    range=x_range, minor=x_minor, ticklabels=(rows[-1] == 4))
            ax.tick_params(which='both', direction='out')
            if i_col > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(r"$\Delta_2/\alpha$", labelpad=ylabelpad)
            labels = [
                #      w_c   w_2     label pos
                ("", (6.20, 5.90 ), (6.35, 5.95), 'FireBrick')
            ]

            for (label, x_y_data, x_y_label, color) in labels:
                ax.scatter(
                    (DeltaC(x_y_data[0]),), (Delta2(x_y_data[1]), ),
                    color=color, marker='x')
                ax.annotate(
                    label, (DeltaC(x_y_label[0]), Delta2(x_y_label[1])),
                    color=color)
            if (i_col == 2):
                fig.text(
                    (left_margin+i_col*(w+hgap)+0.95*w)/fig_width,
                    (bottom_margin+(n_rows-i_row)*(h+vgap)+0.2)/fig_height,
                    r'combined', verticalalignment='bottom',
                    horizontalalignment='right', size=10, color='white')

        # x label for whatever is the bottom row
        ax.set_xlabel(r"$\Delta_c/g$", labelpad=xlabelpad)

        # time labels at top of figure
        fig.text((left_margin+i_col*(w+hgap)+0.95*w)/fig_width,
                 (bottom_margin+(n_rows-1)*(h+vgap)+h-0.2)/fig_height,
                 r'$T = %d$~ns' % T, verticalalignment='top',
                 horizontalalignment='right', size=10, color='white')

    if OUTFOLDER is not None:
        outfile = os.path.join(OUTFOLDER, outfile)

    fig.savefig(outfile)
    print("written %s" % outfile)
    plt.close(fig)


def generate_weyl_plot(stage_table_200, stage_table_050, stage_table_010, outfile):

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
    weyl_offset_y = -0.5
    weyl_width    = 3.5
    weyl_height   = 2.5

    fig_height = bottom_margin + h + top_margin
    fig_width  = (left_margin + 3*w + 2*hgap + cbar_gap + cbar_width
                  + right_margin)
    fig = new_figure(fig_width, fig_height, style=STYLE)

    data = OrderedDict([
            (200, stage_table_200),
            (50,  stage_table_050),
            (10,  stage_table_010), ])

    for i_col, T in enumerate(data.keys()):

        # collection OCT data
        stage_table = data[T]
        stage_table = stage_table[stage_table['category'].str.contains('1freq')]
        (__, t_PE), __ = stage_table.groupby('target', sort=True)
        t_PE_weyl = t_PE[(t_PE['max loss']<0.1) & (t_PE['C']==1.0)]

        weyl = QDYN.weyl.WeylChamber()
        weyl.PE_edge_fg_properties = {
                  'color':'DarkMagenta', 'linestyle':'-', 'lw':0.7}
        weyl.PE_edge_bg_properties = {
                  'color':'DarkMagenta', 'linestyle':'--', 'lw':0.7}
        weyl.weyl_edge_fg_properties = {
                  'color':'Gray', 'linestyle':'-', 'lw':0.5}
        weyl.weyl_edge_bg_properties = {
                  'color':'Gray', 'linestyle':'--', 'lw':0.5}
        pos = [(left_margin+i_col*(w+hgap)+weyl_offset_x)/fig_width,
               (bottom_margin+weyl_offset_y)/fig_height,
               weyl_width/fig_width, weyl_height/fig_height]
        ax_weyl = fig.add_axes(pos, projection='3d');
        weyl.scatter(t_PE_weyl['c1'], t_PE_weyl['c2'], t_PE_weyl['c3'],
                     c='blue', s=5, linewidth=0)
        weyl.labels = {
            'A_1' : weyl.A1 + np.array((-0.07, 0.01 , 0.00)),
            'A_2' : weyl.A2 + np.array((0.01, 0, -0.01)),
            'A_3' : weyl.A3 + np.array((-0.01, 0, 0)),
            'O'   : weyl.O  + np.array((-0.025,  0.0, 0.04)),
            'L'   : weyl.L  + np.array((-0.10, 0, 0.01)),
            'M'   : weyl.M  + np.array((0.05, -0.01, 0)),
            'N'   : weyl.N  + np.array((-0.095, 0, 0.015)),
            'P'   : weyl.P  + np.array((-0.05, 0, 0.008)),
            'Q'   : weyl.Q  + np.array((0, 0.01, 0.03)),
        }
        weyl.render(ax_weyl)
        #for artist in weyl._artists:
            #artist.set_edgecolors = artist.set_facecolors = lambda *args:None
        ax_weyl.xaxis._axinfo['ticklabel']['space_factor'] = 1.0
        ax_weyl.yaxis._axinfo['ticklabel']['space_factor'] = 1.0
        ax_weyl.zaxis._axinfo['ticklabel']['space_factor'] = 1.3
        ax_weyl.xaxis._axinfo['label']['space_factor'] = 1.5
        ax_weyl.yaxis._axinfo['label']['space_factor'] = 1.5
        ax_weyl.zaxis._axinfo['label']['space_factor'] = 1.5
        ax_weyl.xaxis.set_major_formatter(weyl_x_tick_fmt)
        ax_weyl.yaxis.set_major_formatter(weyl_y_tick_fmt)
        ax_weyl.zaxis.set_major_formatter(weyl_z_tick_fmt)

        fig.text((left_margin+i_col*(w+hgap)+0.95*w)/fig_width,
                 (bottom_margin+0*(h+vgap)+h-0.2)/fig_height,
                 r'$T = %d$~ns' % T, verticalalignment='top',
                 horizontalalignment='right', size=10, color='black')

    if OUTFOLDER is not None:
        outfile = os.path.join(OUTFOLDER, outfile)

    fig.savefig(outfile)
    print("written %s" % outfile)
    plt.close(fig)


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
    #                   minimum error   min err (X)     PE error     err(H1)
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
    # * eps_avg^{H1,X}: See notebook QSL_Hadamard1.ipynb,
    #   runfolders in ./QSL_H1_prop. For 50ns, run in
    #   ./propagate_universal/rho/H_L/
    # * For eps_avg^{PE}: See notebook Stage3Analysis.ipynb,
    #   runfolders in ./liouville_prop/stage3/
    # * For eps_avg^{0,X}: See notebook UniversalPropLiouville.ipynb
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
    ax.plot(T, eps_0B, label=r'$\varepsilon_{\text{avg}}^{0,\text{X}}$',
            marker='o', color=get_color('blue'), dashes=ls['dashed'])
    ax.plot(T, eps_H1, label=r'$\varepsilon_{\text{avg}}^{\text{H1,X}}$',
            marker='o', color=get_color('red'), dashes=ls['long-dashed'])
    ax.legend(loc='upper left')
    ax.annotate(r'$\text{QSL\,}^{\text{X}}_{\text{H1}}$',
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
    # variations for talks:
    #generate_field_free_plot(
        #zeta_table, T=50, outfile='fig1_a.pdf', cols=(1, ))
    #generate_field_free_plot(
        #zeta_table, T=50, outfile='fig1_b.pdf', cols=(2, ))
    #generate_field_free_plot(
        #zeta_table, T=50, outfile='fig1_c.pdf', cols=(3, ))

    # Fig 2
    generate_map_plot_combined(stage_table_200, stage_table_050,
                               stage_table_010, zeta_table,
                               outfile='fig2_main.pdf')
    # variations for talks:
    #generate_map_plot_combined(stage_table_200, stage_table_050,
                               #stage_table_010, zeta_table,
                               #outfile='fig2_ac.pdf', rows=(1, ))
    #generate_map_plot_combined(stage_table_200, stage_table_050,
                               #stage_table_010, zeta_table,
                               #outfile='fig2_af.pdf', rows=(1, 2))
    #generate_map_plot_combined(stage_table_200, stage_table_050,
                               #stage_table_010, zeta_table,
                               #outfile='fig2_ac_gi.pdf', rows=(1, 3))
    #generate_map_plot_combined(stage_table_200, stage_table_050,
                               #stage_table_010, zeta_table,
                               #outfile='fig2_gi.pdf', rows=(3, ))

    # Fig 3
    generate_weyl_plot(
        stage_table_200, stage_table_050, stage_table_010,
        outfile='fig3_main.pdf')

    # Fig 4
    generate_error_plot(outfile='fig4_main.pdf')

    # Fig 5
    print("Figure 5 is generated by script in the 'revision' repository")

if __name__ == "__main__":
    sys.exit(main())
