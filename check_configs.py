#!/usr/bin/env python

import sys
import re
import logging
import click
from click import echo


def check_configfile(configfile, verbose=True):
    logger = logging.getLogger()
    logger.debug("Checking %s", configfile)
    m_root = re.search('runs_(?P<T>\d+)_RWA/', configfile)
    if m_root:
        T = int(m_root.group('T'))
        logger.debug("Found root T = %s", str(T))
        with click.open_file(configfile) as in_fh:
            for line in in_fh:
                logger.debug("line: %s", line.strip())
                m_line = re.search('t_stop\s*=\s*(?P<T>[\d.]+)_ns', line)
                if m_line:
                    logger.debug("line contains t_stop")
                    T_in_config = int(float(m_line.group('T')))
                    if T != T_in_config:
                        echo("%s: t_stop = %s does not match T = %s in root"
                             % (configfile, str(T_in_config), str(T)),
                             err=True)
                    else:
                        if verbose:
                            echo("%s: OK" % configfile)
                    return
            # we should not reach this point!
            logger.error("File %s does not contain t_stop", configfile)
    else:
        if verbose:
            echo("%s: skipped" % configfile, err=True)



@click.command()
@click.option('--verbose', '-v', is_flag=True,
    help='Print verbose messages')
@click.option('--debug', is_flag=True,
    help='Log debug information')
@click.argument('list_of_configs', type=click.File('rb'))
def check_configs(list_of_configs, verbose=False, debug=False):
    """Ensure that for all of the listed config files, the pulse duration
    matches that of the root run folder the config file is located in.
    This catches if we screwed up an rsync, overwriting runs_010_RWA with
    runs_100_RWA, for example."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    if debug:
        logger.setLevel(logging.DEBUG)
    for line in list_of_configs:
        check_configfile(line.strip(), verbose)

if __name__ == "__main__":
    sys.exit(check_configs())
