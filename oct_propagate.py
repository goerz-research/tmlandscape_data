import time
import os
import subprocess as sp
import QDYN
import logging
import numpy as np
import re
from stage2_simplex import get_temp_runfolder
from QDYN.pulse import Pulse
from analytical_pulses import AnalyticalPulse
from notebook_utils import pulse_config_compat, ensure_ham_files
from clusterjob.utils import read_file


def propagate(runfolder, pulse_file, rwa, rho=False, rho_pop_plot=False,
        n_qubit=None, n_cavity=None, dissipation=True, keep=False, force=False,
        target='target_gate.dat', use_threads=False):
    """Map runfolder -> 2QGate, by propagating the given pulse file, or reading
    from an existing U.dat (if force is False)

    The file indicated by `pulse_file` may contain either a numerical pulse,
    or, if it has a 'json' extension, and analytic pulse.

    If `rho` is True, propagate in Liouville space. Note that in this case, the
    returned 2QGate is a unitary approximation of the dynamics. The exact value
    of F_avg should be extracted from the 'prop.log' file resulting from the
    propagation. The `rho_pop_plot` option may be used (in combination with
    rho=True and keep=True) to produce the files necessary for a population
    plot. Note that with this option, the propage average gate fidelity can not
    be extracted from the prop.log file.

    By giving the n_qubit and/or n_cavity options, the number of qubit and
    cavity levels can be changed from the value given in the config file.

    If `dissipation` is given as False, the corresponding `dissipation` value
    is set to False in the config, file, resulting in unitary dynamics.

    If `keep` is True, keep all files resulting from the propagation in the
    runfolder. Otherwise, only prop.log and U.dat will be kept. Note that
    the original config file is never overwritten, but the config file from the
    temporary propagation folder (which was possibly modified due to the
    `pulse_file`, `rho_pop_plot`, `n_qubit`, or `n_cavity` options) is copied
    in as ``config.prop``, if `keep` is True.

    If `keep` is None, no files in the runfolder will be modified.

    If use_threads is True, use 4 OpenMP threads
    """
    logger = logging.getLogger(__name__)
    if n_qubit is not None:
        n_qubit = int(n_qubit)
        assert n_qubit > 2
    if n_cavity is not None:
        n_cavity = int(n_cavity)
        assert n_cavity > 1
    gatefile = os.path.join(runfolder, 'U.dat')
    config = os.path.join(runfolder, 'config')
    target_gate_dat = os.path.join(runfolder, target)
    if os.path.isfile(gatefile) and force:
        try:
            os.unlink(gatefile)
        except OSError:
            pass
    U = None
    if (not os.path.isfile(gatefile)) or force:
        temp_config = None
        try:
            temp_runfolder = get_temp_runfolder(runfolder)
            temp_config = os.path.join(temp_runfolder, 'config')
            temp_U_dat = os.path.join(temp_runfolder, 'U.dat')
            logger.debug("Prepararing temp_runfolder %s", temp_runfolder)
            QDYN.shutil.mkdir(temp_runfolder)
            if os.path.isfile(target_gate_dat):
                QDYN.shutil.copy(target_gate_dat, temp_runfolder)
            # copy over the pulse
            analytical_pulse = None
            if pulse_file.endswith(".json"):
                analytical_pulse = AnalyticalPulse.read(
                                   os.path.join(runfolder, pulse_file))
                pulse = analytical_pulse.pulse()
            else:
                pulse = Pulse(os.path.join(runfolder, pulse_file))
            pulse.write(os.path.join(temp_runfolder, 'pulse_prop.dat'))
            # copy over the config file, with modifications
            write_prop_config(config, temp_config, 'pulse_prop.dat',
                              rho=rho, rho_pop_plot=rho_pop_plot,
                              n_qubit=n_qubit, n_cavity=n_cavity,
                              dissipation=dissipation)
            if analytical_pulse is not None:
                pulse_config_compat(analytical_pulse, temp_config,
                                    adapt_config=True)
            logger.info("Propagating %s", runfolder)
            env = os.environ.copy()
            env['OMP_NUM_THREADS'] = '1'
            if use_threads:
                env['OMP_NUM_THREADS'] = '4'
            start = time.time()
            prop_log = os.path.join(runfolder, 'prop.log')
            if keep is None:
                prop_log = os.path.join(temp_runfolder, 'prop.log')
            with open(prop_log, 'w', 0) as stdout:
                ensure_ham_files(temp_runfolder, rwa, stdout, rho, dissipation)
                cmds = []
                if rho:
                    cmds.append(['tm_en_rho_prop', '.'])
                else:
                    cmds.append(['tm_en_prop', '.'])
                for cmd in cmds:
                    stdout.write("**** " + " ".join(cmd) +"\n")
                    sp.call(cmd , cwd=temp_runfolder, env=env,
                            stderr=sp.STDOUT, stdout=stdout)
            U = QDYN.gate2q.Gate2Q(file=temp_U_dat)
            if keep is not None:
                QDYN.shutil.copy(temp_U_dat, runfolder)
            end = time.time()
            logger.info("Finished propagating %s (%d seconds)",
                         runfolder, end-start)
        except Exception as e:
            logger.error(e)
        finally:
            if keep is None:
                keep = False
            if keep:
                if temp_config is not None and os.path.isfile(temp_config):
                    os.rename(temp_config, temp_config + ".prop")
                sp.call(['rsync', '-a', '%s/'%temp_runfolder, runfolder])
            QDYN.shutil.rmtree(temp_runfolder)
    else:
        logger.info("Propagating of %s skipped (gatefile already exists)",
                     runfolder)
    # If we performed a propagation, U should already have been set above based
    # on "U.dat" in the temporary folder. If we're loading from an existing
    # U.dat intead, we do that here.
    if U is None:
        try:
            U = QDYN.gate2q.Gate2Q(file=gatefile)
        except (IOError, OSError) as e:
            logger.error(e)
    if np.isnan(U).any():
        logger.error("gate %s contains NaN", gatefile)
    return U


def write_prop_config(template, config, pulse_file, rho=False,
        rho_pop_plot=False, n_qubit=None, n_cavity=None, dissipation=None):
    """Write a new config file based on template, updated to propagate the
    numerical pulse stored in `pulse_file`.
    """
    with open(template, 'r') as in_fh, open(config, 'w') as out_fh:
        section = ''
        for line in in_fh:
            m = re.match(r'^\s*(?P<section>[a-z_]+)\s*:', line)
            if m:
                section = m.group('section')
            if section == 'pulse':
                line = re.sub(r'type\s*=\s*\w+', r'type = file', line)
                line = re.sub(r'filename\s*=\s*[\w.]+',
                              r'filename = %s'%(pulse_file), line)
            elif section == 'user_logicals':
                line = re.sub(r'prop_guess\s*=\s*(T|F)',
                              r'prop_guess = T', line)
            if rho_pop_plot:
                line = re.sub(
                    r'rho_prop_mode\s*=\s*(full|pop_dynamics)',
                    r'rho_prop_mode = pop_dynamics', line)
            else:
                line = re.sub(
                    r'rho_prop_mode\s*=\s*(full|pop_dynamics)',
                    r'rho_prop_mode = full', line)
            if n_qubit is not None:
                line = re.sub(r'n_qubit\s*=\s*\d+',
                            r'n_qubit = %d' % n_qubit, line)
            if n_cavity is not None:
                line = re.sub(r'n_cavity\s*=\s*\d+',
                            r'n_cavity = %d' % n_cavity, line)
            if dissipation is not None:
                if dissipation:
                    val = 'T'
                    line = re.sub(r'prop\s*=\s*cheby', r'prop = newton', line)
                else:
                    val = 'F'
                    line = re.sub(r'prop\s*=\s*newton', r'prop = cheby', line)
                line = re.sub(r'dissipation\s*=\s*[TF]',
                              r'dissipation = %s' % val, line)
            out_fh.write(line)
    config_content = read_file(config)
    if rho:
        assert "rho_prop_mode" in config_content
        assert "gamma_phi_1" in config_content
        assert "gamma_phi_2" in config_content
