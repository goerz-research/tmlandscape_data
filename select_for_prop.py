#!/usr/bin/env python
"""copy stage 4 runfolders, in preparation for propagation"""
import os
import re
import sys
import logging
import filecmp
import QDYN
import numpy as np
from notebook_utils import find_folders, find_leaf_folders
from clusterjob.utils import read_file

def get_stage4_runfolders(runs, stage4_folder):
    """Return a list of all stage 4 runfolders that contain the file pulse.dat
    (result of OCT)
    """
    runfolders = []
    for folder in find_folders(runs, stage4_folder):
        for subfolder in find_leaf_folders(folder):
            if os.path.isfile(os.path.join(subfolder, 'pulse.dat')):
                runfolders.append(subfolder)
    return runfolders


def prepare_prop(oct_folder, prop_folder, dry_run=False):
    """Generate propagation folder based on the given stage 4 runfolder.

    The oct_folder must contain a config file and a file pulse.dat that
    is the result of running OCT. The config file will be modified such that it
    is valid both for Liouville and Hilbert space propagation.
    """
    logger = logging.getLogger(__name__)
    stage4_guess_file = os.path.join(oct_folder, 'pulse.guess')
    stage4_pulse_file = os.path.join(oct_folder, 'pulse.dat')
    stage4_target_gate = os.path.join(oct_folder, 'target_gate.dat')
    stage4_config = os.path.join(oct_folder, 'config')
    assert prop_folder != oct_folder
    prop_pulse_file = os.path.join(prop_folder, 'pulse.dat')
    prop_config = os.path.join(prop_folder, 'config')
    prop_U_dat = os.path.join(prop_folder, 'U.dat')

    if os.path.isdir(prop_folder):
        if os.path.isfile(prop_pulse_file):
            # we discard any existing data that does not match the pulse in the
            # oct_folder
            if not filecmp.cmp(prop_pulse_file, stage4_pulse_file):
                logger.debug("Removing %s: stale pulse" % prop_folder)
                QDYN.shutil.rmtree(prop_folder, ignore_errors=True)
        if os.path.isfile(prop_U_dat):
            logger.debug("Skipping %s: contains U.dat" % prop_folder)
        else:
            # If the runfolder doesn't contain the result of a propagation, we
            # might was well get rid of it and start afresh
            logger.debug("Removing %s: incomplete" % prop_folder)
            QDYN.shutil.rmtree(prop_folder, ignore_errors=True)

    if not os.path.isdir(prop_folder):

        msg = "Create %s" % prop_folder
        if not dry_run:
            logger.debug(msg)
            QDYN.shutil.mkdir(prop_folder)
        else:
            print(msg)

        msg = "Copy %s, %s, %s -> %s" \
              % (stage4_pulse_file, stage4_guess_file, stage4_target_gate,
                 prop_folder)
        if not dry_run:
            logger.debug(msg)
            QDYN.shutil.copy(stage4_pulse_file, prop_folder)
            QDYN.shutil.copy(stage4_guess_file, prop_folder)
            QDYN.shutil.copy(stage4_target_gate, prop_folder)
        else:
            print("  "+msg)

        msg = "Transfer %s -> %s" % (stage4_config, prop_config)
        if not dry_run:
            logger.debug(msg)
            config_content = read_file(stage4_config)
            with \
            open(stage4_config, 'r') as in_fh, \
            open(prop_config, 'w') as out_fh:
                for line in in_fh:
                    if 'prop_guess' in line:
                        line = re.sub(r'/prop_guess\s*=\s*[T|F]',
                                      r'prop_guess = F', line)
                    # the following two cases are for making the propagation
                    # work for Liouville space as well
                    if 'user_strings:' in line:
                        if not 'rho_prop_mode' in line:
                            line = line.strip()+", rho_prop_mode = full\n"
                    if 'gamma_2 =' in line:
                        if not 'gamma_phi_1' in config_content:
                            if not line.endswith(", &\n"):
                                line = line.strip() + ", &\n"
                            line += "gamma_phi_1 = 0.0, &\n" \
                                    "gamma_phi_2 = 0.0, &\n"
                    out_fh.write(line)
            # Ensure that the config file has the required settings for
            # Liouville space.
            config_content = read_file(prop_config)
            assert "rho_prop_mode = full" in config_content
            assert "gamma_phi_1 = 0.0" in config_content
            assert "gamma_phi_2 = 0.0" in config_content
        else:
            print("  "+msg)


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
    arg_parser.add_option(
        '--debug', action='store_true', dest='debug',
        default=False, help="Enable debugging output")
    arg_parser.add_option(
        '--stage4-folder', action='store', dest='stage4_folder',
        default='stage4', help="Name of stage 4 folder. Defaults to 'stage4'")
    arg_parser.add_option(
        '--stage-prop-folder', action='store', dest='stage_prop_folder',
        default='stage_prop', help="Name of propagation stage folder. Defaults "
        "to 'stage_prop'")
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
    logger = logging.getLogger()
    if options.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    folders = get_stage4_runfolders(runs, stage4_folder=options.stage4_folder)
    for oct_folder in folders:
        prop_folder = runfolder.replace(options.stage4_folder,
                                        options.stage_prop_folder)
        prepare_prop(oct_folder, prop_folder, dry_run=options.dry_run)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
