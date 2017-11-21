#!/bin/bash

if [ $1 ] && [ $1 = '-c' ] # clean
then
    rm -rf *.log save_model build
fi

mkdir -p build
cd build
cmake ..
make -j $(nproc)
