#!/bin/bash
conda create -p ./venv python=2.7 ipython=4.0.0 ipython-notebook numpy=1.9.3 scipy=0.16.0 matplotlib=1.4.3 bokeh=0.10.0 sympy=0.7.6.1 pandas=0.16.2 paramiko click=4.1 psutil pexpect

wget https://raw.githubusercontent.com/goerz/mplstyles/master/interactive.mplstyle -O ./venv/lib/python2.7/site-packages/matplotlib/mpl-data/matplotlibrc

source setenv.sh
mkdir -p ./venv/src

(cd $PREFIX/src/ && git clone git@jerusalem.physik.uni-kassel.de:qdyn)
(cd $PREFIX/src/ && git clone git@jerusalem.physik.uni-kassel.de:qdynpylib)
(cd $PREFIX/src/ && git clone git@jerusalem.physik.uni-kassel.de:goerz/transmon_oct)
(cd $PREFIX/src/ && git clone git@jerusalem.physik.uni-kassel.de:goerz/QDYNTransmonLib)

(cd $PREFIX/bin/ && ln -s ../../rewrite_dissipation.py)

(cd $PREFIX/src/qdyn && git checkout 50b2d685df2070f8ff93d299e6b374b322a10cb7 && ./configure --prefix=$PREFIX --no-parallel-oct --no-hooks && make install)
(cd $PREFIX/src/qdynpylib && git checkout dd8894d9ef972ef760627c5232cbbd10eebb6d18 && pip install -e .)
(cd $PREFIX/src/transmon_oct && git checkout 7515584cfa43c00b7482fa419a379e0a9ef606c6 && ./configure --prefix=$PREFIX && make install)
(cd $PREFIX/src/QDYNTransmonLib && git checkout master && pip install -e .)

pip install 'mgplottools==1.0.0'
pip install 'fortranfile==0.2.1'
pip install 'clusterjob==1.1.3'
