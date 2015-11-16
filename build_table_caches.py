#!/usr/bin/env python
from __future__ import print_function
import sys
import os
import re
from glob import glob

import QDYN
from notebook_utils import get_stage1_table, get_stage2_table, get_stage3_table
import pandas as pd
pd.options.display.max_colwidth = 512


def check_stage1(stage1_table):
    """Print any runfolder for which there is missing data"""
    for col in ['C', 'avg loss', 'max loss', 'E0 [MHz]', 'J_PE', 'J_SQ']:
        for runfolder in stage1_table[stage1_table[col].isnull()].index:
            print("Runfolder %s: missing value for %s" % (runfolder, col))


def check_oct(stage_table):
    for col in ['C', 'avg loss', 'max loss', 'J_PE', 'J_SQ', 'F_avg', 'c1']:
        for runfolder in stage_table[stage_table[col].isnull()].index:
            print("Runfolder %s: missing value for %s" % (runfolder, col))
    (__, t_PE), (__, t_SQ) = stage_table.groupby('target', sort=True)
    t_PE['PE runfolder'] = t_PE.index
    t_SQ['SQ runfolder'] = t_SQ.index
    Q_table = pd.concat([
         t_PE.rename(columns={'F_avg': 'F_avg (PE)'})
         [['w1 [GHz]', 'w2 [GHz]', 'wc [GHz]', 'category', 'F_avg (PE)',
           'PE runfolder']]
         .set_index(['w1 [GHz]', 'w2 [GHz]', 'wc [GHz]', 'category']),
         t_SQ.rename(columns={'F_avg': 'F_avg (SQ)'})
         [['w1 [GHz]', 'w2 [GHz]', 'wc [GHz]', 'category', 'F_avg (SQ)',
           'SQ runfolder']]
         .set_index(['w1 [GHz]', 'w2 [GHz]', 'wc [GHz]', 'category'])
        ], axis=1).reset_index()
    for col in ['F_avg (PE)', 'F_avg (SQ)']:
        for i in Q_table[Q_table[col].isnull()].index:
            print(("Incomplete Q data w2={w2:g}GHz, wc={wc:g}GHz, cat={cat}: "
                "{col}")
                  .format(w2=Q_table.ix[i]['w2 [GHz]'],
                          wc=Q_table.ix[i]['wc [GHz]'],
                          cat=Q_table.ix[i]['category'],
                          col=col))
            rf_PE = Q_table.ix[i]['PE runfolder']
            rf_SQ = Q_table.ix[i]['SQ runfolder']
            if pd.isnull(rf_PE):
                print("  No PE folder")
            else:
                if os.path.isdir(rf_PE):
                    print("  %s exists" % rf_PE)
                else:
                    print("  %s does not exist" % rf_PE)
            if pd.isnull(rf_SQ):
                print("  No SQ folder")
            else:
                if os.path.isdir(rf_SQ):
                    print("  %s exists" % rf_SQ)
                else:
                    print("  %s does not exist" % rf_SQ)


def collect(reader, checker):
    for stage in reader:
        print("\n*** %s ***" % stage)
        dump_file = stage+'_table.cache'
        for folder in glob("./runs*_RWA"):
            print("processing folder: %s..." % folder)
            table = reader[stage](folder)
            table_file_name = "{stage}_table_{runs}.dat".format(
                    stage=stage, runs=re.sub(r'^.*runs_(.*)$', r'\1', folder))
            with open(table_file_name, 'w') as out_fh:
                out_fh.write(table.to_string())
            print("DONE processing folder: %s..." % folder)
            if stage in checker:
                checker[stage](table)
        reader[stage].dump(dump_file)
        print("Written %s" % dump_file)


def main(argv=None):
    from optparse import OptionParser
    if argv is None:
        argv = sys.argv
    arg_parser = OptionParser(
    usage = "usage: %prog [options]",
    description = __doc__)
    arg_parser.add_option(
        '--check', action='store_true', dest='check',
        default=False, help="Check for missing/invalid data")
    options, args = arg_parser.parse_args(argv)
    reader = {
        'stage1': QDYN.memoize.memoize(get_stage1_table),
        'stage2': QDYN.memoize.memoize(get_stage2_table),
        'stage3': QDYN.memoize.memoize(get_stage3_table)
    }
    if options.check:
        checker = {
            'stage1': check_stage1,
            'stage2': check_oct,
            'stage3': check_oct
        }
    else:
        checker = {}
    collect(reader, checker)

if __name__ == "__main__":
    sys.exit(main())

