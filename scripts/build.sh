#!/usr/bin/bash
# Copyright (c) 2023 LEI WANG
ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

pip install -r $ROOT/requirements.txt

pushd $ROOT/diffusers
pip install -v .
popd