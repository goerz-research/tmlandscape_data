#!/bin/bash

export PREFIX=`pwd`/venv
export PATH=$PREFIX/bin:$PATH
export export LD_LIBRARY_PATH=$PREFIX/lib:$LD_LIBRARY_PATH
./venv/bin/ipython notebook
