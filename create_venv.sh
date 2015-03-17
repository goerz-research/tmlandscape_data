#!/bin/bash
conda create -p ./venv anaconda
./venv/bin/conda install -y -p ./venv pexpect

export PREFIX=`pwd`/venv
export LD_LIBRARY_PATH=$PREFIX/lib:$LD_LIBRARY_PATH
export PATH=$PREFIX/bin:$PATH

mkdir -p ./venv/src

(cd $PREFIX/src/ && git clone git@jerusalem.physik.uni-kassel.de:qdyn)
(cd $PREFIX/src/ && git clone git@jerusalem.physik.uni-kassel.de:qdynpylib)
(cd $PREFIX/src/ && git clone git@jerusalem.physik.uni-kassel.de:goerz/transmon_oct)
(cd $PREFIX/src/ && git clone git@jerusalem.physik.uni-kassel.de:goerz/QDYNTransmonLib)

(cd $PREFIX/src/qdyn && git checkout master && ./configure --prefix=$PREFIX --no-hooks && make install)
(cd $PREFIX/src/qdynpylib && git checkout master && python setup.py install)
(cd $PREFIX/src/transmon_oct && git checkout master && ./configure --prefix=$PREFIX && make install)
(cd $PREFIX/src/QDYNTransmonLib && git checkout master && python setup.py install)

pip install mgplottools
pip install fortranfile
