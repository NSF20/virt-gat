#!/bin/bash
PYTHON_PATH=$HOME/projects/packages/search_quality/python37_gcc482_pd18_pgl12
export LD_LIBRARY_PATH=${PYTHON_PATH}/lib:$LD_LIBRARY_PATH
export PATH=${PYTHON_PATH}/bin:$PATH
alias gccpython="/opt/compiler/gcc-4.8.2/lib/ld-linux-x86-64.so.2 --library-path /opt/compiler/gcc-4.8.2/lib:${PYTHON_PATH}/lib:/usr/lib64:$LD_LIBRARY_PATH ${PYTHON_PATH}/bin/python"
