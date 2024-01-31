#!/usr/bin/bash
# Copyright (c) 2023 LEI WANG
ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

pushd $ROOT/diffusers
python setup.py deveop pip install -e .
popd
