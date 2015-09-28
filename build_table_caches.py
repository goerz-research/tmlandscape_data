#!/usr/bin/env python -u
from __future__ import print_function
from notebook_utils import get_stage1_table, get_stage2_table, get_stage3_table
from glob import glob
import QDYN
reader = {
    'stage1': QDYN.memoize.memoize(get_stage1_table),
    'stage2': QDYN.memoize.memoize(get_stage2_table),
    'stage3': QDYN.memoize.memoize(get_stage3_table)
}
for stage in reader:
    print("\n*** %s ***" % stage)
    dump_file = stage+'_table.cache'
    for folder in glob("./runs*_RWA"):
        print("processing folder: %s..." % folder)
        reader[stage](folder)
        print("DONE processing folder: %s..." % folder)
    reader[stage].dump(dump_file)
    print("Written %s" % dump_file)
