#!/usr/bin/env python
from __future__ import print_function
from notebook_utils import get_stage1_table, get_stage2_table, get_stage3_table
import re
from glob import glob
import QDYN
import pandas as pd
pd.options.display.max_colwidth = 512
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
        table = reader[stage](folder)
        table_file_name = "{stage}_table_{runs}.dat".format(
                stage=stage, runs=re.sub(r'^.*runs_(.*)$', r'\1', folder))
        with open(table_file_name, 'w') as out_fh:
            out_fh.write(table.to_string())
        print("DONE processing folder: %s..." % folder)
    reader[stage].dump(dump_file)
    print("Written %s" % dump_file)
