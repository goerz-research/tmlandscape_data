#!/usr/bin/env python
"""Propagate the given stage2 RWA runfolder in the lab from, producing the files
U_LAB.dat and prop_lab.log In the runfolder. Assumes that the files
pulse_opt.json and config exists"""

from optparse import OptionParser
import logging
logging.basicConfig()
from notebook_utils import prop_LAB
import sys
import os


def main(argv=None):
    """Main routine"""
    if argv is None:
        argv = sys.argv
    arg_parser = OptionParser(
    usage = "%prog [options] <runfolder>",
    description = __doc__)
    arg_parser.add_option(
        '--debug', action='store_true', dest='debug',
        default=False, help="Show debug output")
    options, args = arg_parser.parse_args(argv)
    try:
        runfolder = args[1]
        if not os.path.isdir(runfolder):
            arg_parser.error("runfolder %s does not exist"%runfolder)
    except IndexError:
        arg_parser.error("runfolder be given")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if options.debug:
        logger.setLevel(logging.DEBUG)
    assert 'SCRATCH_ROOT' in os.environ, \
    "SCRATCH_ROOT environment variable must be defined"
    config_file = os.path.join(runfolder, 'config')
    pulse_json = os.path.join(runfolder, 'pulse_opt.json')
    U_LAB_dat = os.path.join(runfolder, 'U_LAB.dat')
    prop_lab_log = os.path.join(runfolder, 'prop_lab.log')
    if os.path.isfile(U_LAB_dat) and os.path.isfile(prop_lab_log):
        logger.debug("Folder %s has already been propagated in the LAB frame",
                     runfolder)
        return 0
    if os.path.isfile(config_file) and os.path.isfile(pulse_json):
        prop_LAB(config_file, pulse_json, outfolder=runfolder, runfolder=None)
    else:
        arg_parser.error("config and pulse_opt.json must exist in runfolder")


if __name__ == "__main__":
    sys.exit(main())
