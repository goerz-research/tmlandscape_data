#!/usr/bin/env python
import os


def decay_qubit(i):
    """Return the imaginary potential in GHz for qubit level i = 0..

       Source: Peterer et al, Phys. Rev. Lett. 114, 010501 (2015).
    """
    if i == 0:
        return 0.0
    elif i == 1:
        # 84 microsec
        return -0.5 * 1.0e-3 * 0.012
    elif i == 2:
        # 41 microsec
        return -0.5 * 1.0e-3 * 0.024
    elif i == 3:
        # 30 microsec
        return -0.5 * 1.0e-3 * 0.033
    else:
        # higher levels are "truncated" with an extremely high decay rate
        return -0.5 * 1.0e-3 * 500


def decay_cavity(n):
    """Return the imaginary potential in GHz for cavity level i = 0..

       Source: A. W. Cross and J. M. Gambetta, Phys. Rev. A 91 032325 (2015)
    """
    if n < 5:
        # 20 microsec
        return n * -0.5 * 1.0e-3 * 0.05
    else:
        # higher levels are "truncated" with an extremely high decay rate
        return -0.5 * 1.0e-3 * 500


os.rename("ham_drift.dat", "ham_drift.bak")
fmt = "%8d%8d%25.16E%25.16E%7d%7d%7d    %7d%7d%7d\n"
with open("ham_drift.bak", "r") as in_fh, open("ham_drift.dat", 'w') as out_fh:
    for line in in_fh:
        if line.startswith("#"):
            out_fh.write(line)
        else:
            row, col, E_re, E_im, i_row, j_row, n_row, i_col, j_col, n_col \
            = line.split()
            row = int(row)
            col = int(col)
            E_re = float(E_re)
            E_im = float(E_im)
            i_row = int(i_row)
            j_row = int(j_row)
            n_row = int(n_row)
            i_col = int(i_col)
            j_col = int(j_col)
            n_col = int(n_col)
            if row == col:
                E_im = decay_qubit(i_row) + decay_qubit(j_row) \
                       + decay_cavity(n_row)
                out_fh.write(fmt % (row, col, E_re, E_im, i_row, j_row, n_row,
                                    i_col, j_col, n_col))
            else:
                out_fh.write(line)

