#!/usr/bin/env python
"""Run stage4 (optimization)

Calls ./run_oct.py for any found runfolder that has 'stage4' in its path,
inside RUNS (as prepared by ./select_for_stage4.py).

Assumes that the runfolders contain the files config, pulse.guess.

Ceates the new files in the runfolders:
oct_iters.dat  oct.log  prop.log  pulse.dat  U.dat

If all the output files already exist, the runfolder is skipped unless
the --continue option is given.
"""
import sys
import run_stage3
run_stage3.STAGE = 'stage4'
run_stage3.__doc__ = __doc__

if __name__ == "__main__":
    sys.exit(run_stage3.main())
