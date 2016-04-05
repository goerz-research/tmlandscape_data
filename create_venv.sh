#!/bin/bash
conda create -p ./venv python=2.7 ipython=4.0.0 ipython-notebook numpy=1.9.3 scipy=0.17.0 matplotlib=1.4.3 bokeh=0.10.0 sympy=0.7.6.1 pandas=0.16.2 paramiko click=4.1 ipywidgets psutil pexpect sh cython

wget https://raw.githubusercontent.com/goerz/mplstyles/master/interactive.mplstyle -O ./venv/lib/python2.7/site-packages/matplotlib/mpl-data/matplotlibrc

source setenv.sh
mkdir -p ./venv/src

(cd $PREFIX/src/ && git clone git@jerusalem.physik.uni-kassel.de:qdyn)
(cd $PREFIX/src/ && git clone git@jerusalem.physik.uni-kassel.de:qdynpylib)
(cd $PREFIX/src/ && git clone git@jerusalem.physik.uni-kassel.de:goerz/transmon_oct)
(cd $PREFIX/src/ && git clone git@jerusalem.physik.uni-kassel.de:goerz/QDYNTransmonLib)
(cd $PREFIX/src/ && git clone git://github.com/goerz/clusterjob.git)

(cd $PREFIX/bin/ && ln -s ../../rewrite_dissipation.py)

(cd $PREFIX/src/qdyn && git checkout 50b2d685df2070f8ff93d299e6b374b322a10cb7 && ./configure --prefix=$PREFIX --no-hooks && make install)
(cd $PREFIX/src/qdynpylib && git checkout 1d9255d2c55dac4e39b9c25701d73935a16cc3fe && pip install -e .)
(cd $PREFIX/src/transmon_oct && git checkout 03508eb57aa9da474d018ba0a80756003104b893 && ./configure --prefix=$PREFIX && make install)
(cd $PREFIX/src/QDYNTransmonLib && git checkout master && pip install -e .)
(cd $PREFIX/src/clusterjob && git checkout cd01f3ce9b798b418fb0cfe3c45fc319ff4e90cf && pip install -e .)

pip install 'mgplottools==1.0.0'
pip install 'fortranfile==0.2.1'
pip install --no-use-wheel qutip
