#!/usr/bin/env python
import sys
import run_stage3

run_stage3.STAGE = 'stage4'

if __name__ == "__main__":
    sys.exit(run_stage3.main())
