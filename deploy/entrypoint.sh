#!/bin/sh

poetry env list

export POETRY_VIRTUALENVS_IN_PROJECT=true
source "$(poetry env info --path)/bin/activate"

python3 -u train/train.py 2>&1 | tee data/model/console.log
