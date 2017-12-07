#!/bin/bash

if [ $1 ] && [ $1 = '-c' ] # clean
then
    rm -rf *.log save_model build
fi

mkdir -p build
cd build
cmake ..

if [[ "$OSTYPE" == "linux-gnu" ]]; then
    # Linux
    make -j `nproc`
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # Mac OSX
    make -j `sysctl -n hw.ncpu`
fi
