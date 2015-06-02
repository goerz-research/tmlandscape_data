#!/bin/bash
conda create -p ./venv ipython ipython-notebook numpy scipy matplotlib bokeh pexpect sympy pandas

wget https://raw.githubusercontent.com/goerz/mplstyles/master/interactive.mplstyle -O ./venv/lib/python2.7/site-packages/matplotlib/mpl-data/matplotlibrc

source setenv.sh
mkdir -p ./venv/src

(cd $PREFIX/src/ && git clone git@jerusalem.physik.uni-kassel.de:qdyn)
(cd $PREFIX/src/ && git clone git@jerusalem.physik.uni-kassel.de:qdynpylib)
(cd $PREFIX/src/ && git clone git@jerusalem.physik.uni-kassel.de:goerz/transmon_oct)
(cd $PREFIX/src/ && git clone git@jerusalem.physik.uni-kassel.de:goerz/QDYNTransmonLib)

(cd $PREFIX/bin/ && ln -s ../../rewrite_dissipation.py)

(cd $PREFIX/src/qdyn && git checkout master && ./configure --prefix=$PREFIX --no-hooks && make install)
(cd $PREFIX/src/qdynpylib && git checkout master && pip install -e .)
(cd $PREFIX/src/transmon_oct && git checkout master && ./configure --prefix=$PREFIX && make install)
(cd $PREFIX/src/QDYNTransmonLib && git checkout master && pip install -e .)

pip install mgplottools
pip install fortranfile
pip install clusterjob
