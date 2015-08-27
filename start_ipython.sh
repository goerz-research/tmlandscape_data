#!/bin/bash

source setenv.sh
./venv/bin/ipython notebook --ip=* --certfile=mycert.pem
