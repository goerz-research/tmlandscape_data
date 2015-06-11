#!/usr/bin/env python
import os
import sys
import QDYN
from pre_simplex_scan import runfolder_to_params
from notebook_utils import find_files, find_folders
from analytical_pulses import AnalyticalPulse

def select_runs(stage1_folder):
    selected = {
        'PE_1freq_center': None,
        'PE_1freq_random': None,
        'PE_2freq_resonant': None,
        'PE_2freq_random': None,
        'PE_5freq_random': None,
        'SQ_1freq_center': None,
        'SQ_1freq_random': None,
        'SQ_2freq_resonant': None,
        'SQ_2freq_random': None,
        'SQ_5freq_random': None,
    }
    w_2_selection = None
    w_c_selection = None

    for gatefile in find_files(stage1_folder, 'U.dat'):

        # get the relevant parameters
        runfolder, U_dat = os.path.split(gatefile)
        w_2, w_c, E0, pulse_label = runfolder_to_params(runfolder)
        if w_2_selection is None:
            w_2_selection = w_2
        else:
            assert(w_2_selection == w_2), \
            "All subfolders must be for the same qubit parameters"
        if w_c_selection is None:
            w_c_selection = w_c
        else:
            assert(w_c_selection == w_c), \
            "All subfolders must be for the same qubit parameters"
        U = QDYN.gate2q.Gate2Q(file=gatefile)
        C = U.closest_unitary().concurrence()
        loss = U.pop_loss()
        pulse = AnalyticalPulse.read(os.path.join(runfolder, 'pulse.json'))

        # The exact resonant result returned broken results (norm increase,
        # probably due to bug in choice of logical eigenstates). We skip it
        if w_2 == 6000:
            return None

        # select the relevant category (key in `selected`)
        if '1freq' in pulse_label:
            if '1freq_center' in pulse_label:
                category = '1freq_center'
            else:
                category = '1freq_random'
        elif '2freq' in pulse_label:
            if '2freq_resonant' in pulse_label:
                category = '2freq_resonant'
            else:
                category = '2freq_random'
            pass
        elif '5freq' in pulse_label:
            category = '5freq_random'

        if loss > 0.1:
            # we disregard any guess that leads to more than 10% loss
            continue

        # use with regard to perfect entangler
        for target, selector in [('PE', lambda C, C_prev: C > C_prev),
                                 ('SQ', lambda C, C_prev: C < C_prev)]:
            target_category = target+'_'+category
            if selected[target_category] is None:
                selected[target_category] = (C, loss, E0, pulse, runfolder)
            else:
                C_prev, loss_prev, E0_prev, __, __ = selected[target_category]
                rel_diff = abs(C-C_prev) / abs(C)
                if rel_diff <= 0.01:
                    # all things being equal (i.e. less than 1% change), we
                    # prefer pulses to be around 100 MHz amplitude -- not field
                    # free (would lead to noisy optimized pulses later), and
                    # not too large (optimization will generally increase
                    # amplitude even further)
                    if (abs(E0-100.0) < abs(E0_prev-100.0)):
                        selected[target_category] \
                        = (C, loss, E0, pulse, runfolder)
                else:
                    if selector(C, C_prev):
                        selected[target_category] \
                        = (C, loss, E0, pulse, runfolder)

    if None in selected.values():
        print("Could not find data for all categories. Please debug.")
        from IPython.core.debugger import Tracer; Tracer()()
    return w_2_selection, w_c_selection, selected


def all_select_runs():
    """Analyze the runfolders generated by run_state1.py, select the best runs
    in preparation for stage 2"""
    results = []
    for stage1_folder in list(find_folders("runs", "stage1")):
        selection = select_runs(stage1_folder)
        if selection is not None:
            results.append(selection)
    return results


def main():
    """Main routine"""
    all_select_runs()


if __name__ == "__main__":
    sys.exit(main())