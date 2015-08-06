#!/bin/bash
conda create -p ./venv ipython ipython-notebook numpy scipy matplotlib bokeh pexpect sympy pandas paramiko

wget https://raw.githubusercontent.com/goerz/mplstyles/master/interactive.mplstyle -O ./venv/lib/python2.7/site-packages/matplotlib/mpl-data/matplotlibrc

source setenv.sh
mkdir -p ./venv/src

(cd $PREFIX/src/ && git clone git@jerusalem.physik.uni-kassel.de:qdyn)
(cd $PREFIX/src/ && git clone git@jerusalem.physik.uni-kassel.de:qdynpylib)
(cd $PREFIX/src/ && git clone git@jerusalem.physik.uni-kassel.de:goerz/transmon_oct)
(cd $PREFIX/src/ && git clone git@jerusalem.physik.uni-kassel.de:goerz/QDYNTransmonLib)

(cd $PREFIX/bin/ && ln -s ../../rewrite_dissipation.py)

(cd $PREFIX/src/qdyn && git checkout 28d4b4650ec062f00c8ff11179aa276a5cffa9bc && ./configure --prefix=$PREFIX --no-hooks && make install)
(cd $PREFIX/src/qdynpylib && git checkout master && pip install -e .)
(cd $PREFIX/src/transmon_oct && git checkout 7515584cfa43c00b7482fa419a379e0a9ef606c6 && ./configure --prefix=$PREFIX && make install)
(cd $PREFIX/src/QDYNTransmonLib && git checkout master && pip install -e .)

pip install mgplottools
pip install fortranfile
pip install clusterjob
