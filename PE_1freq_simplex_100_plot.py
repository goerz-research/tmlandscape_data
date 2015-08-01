#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for generating a plot of gate error (data included in file)
"""
import os
import sys
import matplotlib
matplotlib.use('Agg')
from notebook_utils import get_stage2_table, PlotGrid
from QDYN.memoize import memoize

STYLE = 'slides.mplstyle'


def create_figure(outfile, table):

    plots = PlotGrid(publication=True)

    plots.add_cell(table['w2 [GHz]'], table['wc [GHz]'], table['C'],
                    vmin=0.0, vmax=1.0, contour_levels=11,
                    title='concurrence')
    plots.add_cell(table['w2 [GHz]'], table['wc [GHz]'], table['max loss'],
                    vmin=0.0, vmax=1.0,
                    contour_levels=11,
                    title='max. pop. loss')

    # output
    fig = plots.plot(quiet=True, show=False, style=STYLE)
    fig.savefig(outfile, format=os.path.splitext(outfile)[1][1:])


@memoize
def read_data():
    category = '1freq_center'
    full_table = get_stage2_table('./runs_100_RWA')
    (__, t_PE), (__, t_SQ) = full_table.groupby('target', sort=True)
    target_table = t_PE
    table_grouped = target_table.groupby('category')
    return table_grouped.get_group(category)


def main(argv=None):
    if argv is None:
        argv = sys.argv
    basename = os.path.splitext(__file__)[0]
    outfile = basename + '.png'
    read_data.load(basename + '.cache')
    table = read_data()
    read_data.dump(basename + '.cache')
    create_figure(outfile, table)


if __name__ == "__main__":
    sys.exit(main())

