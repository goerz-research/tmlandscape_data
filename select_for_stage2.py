#!/usr/bin/env python
"""Select best runs from stage 1, create stage 2 runfolders"""
import os
import sys
import logging
from QDYN import shutil
import filecmp

def select_for_stage2(stage1_table, target='PE'):
    """For each choice of (w2, wc, category), keep only the row that best
    fulfills the target (PE or SQ)

    stage1_table must contain the columns 'w_2 [GHz]', 'w_c [GHz]', 'category',
    'J_PE', and 'J_SQ', and use the stage1 runfolder as an index
    """
    field_filter = stage1_table['category'] != 'field_free'

    return stage1_table[field_filter]\
        .groupby(['w2 [GHz]', 'wc [GHz]', 'category'], as_index=False)\
        .apply(lambda df: df[df.index==df['J_%s'%target].idxmin()])\
        .reset_index(level=0,drop=True)


def all_select_runs(runs, dry_run=False):
    """Analyze the runfolders generated by run_state1.py, select the best runs
    in preparation for stage 2"""
    from notebook_utils import get_stage1_table
    from os.path import join, isfile
    logger = logging.getLogger(__name__)
    stage1_table = get_stage1_table(runs)
    for target in ['PE', 'SQ']:
        table = select_for_stage2(stage1_table,  target)
        for stage1_runfolder, row in table.iterrows():
            stage2_runfolder = os.path.join(runs,
                              'w2_%dMHz_wc_%dMHz' % (row['w2 [GHz]']*1000,
                                                     row['wc [GHz]']*1000),
                              'stage2',
                              '%s_%s' % (target, row['category']))
            if isfile(join(stage2_runfolder, 'pulse.json')):
                if (not filecmp.cmp(join(stage1_runfolder, 'pulse.json'),
                                    join(stage2_runfolder, 'pulse.json'))):
                    logger.warn("%s does not match %s",
                                join(stage1_runfolder, 'pulse.json'),
                                join(stage2_runfolder, 'pulse.json'))
            else:
                shutil.mkdir(stage2_runfolder)
                for file in ['config', 'pulse.json']:
                    if dry_run:
                        print "Copy %s -> %s" % (
                            os.path.join(stage1_runfolder, file),
                            os.path.join(stage2_runfolder, file))
                    else:
                        shutil.copy(os.path.join(stage1_runfolder, file),
                                    os.path.join(stage2_runfolder, file))


def main(argv=None):
    """Main routine"""
    from optparse import OptionParser
    if argv is None:
        argv = sys.argv
    arg_parser = OptionParser(
    usage = "usage: %prog [options] RUNS",
    description = __doc__)
    arg_parser.add_option(
        '-n', action='store_true', dest='dry_run',
        default=False, help="Perform dry-run")
    options, args = arg_parser.parse_args(argv)
    try:
        runs = os.path.join('.', os.path.normpath(args[1]))
    except IndexError:
        arg_parser.error("You must give RUNS")
    if not os.path.isdir(runs):
        arg_parser.error("RUNS must be a folder (%s)" % runs)
    if not runs.startswith(r'./'):
        arg_parser.error('RUNS must be relative to current folder, '
                         'e.g. ./runs')
    all_select_runs(runs, dry_run=options.dry_run)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
