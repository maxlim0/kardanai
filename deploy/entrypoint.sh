#!/bin/sh

poetry env list

export POETRY_VIRTUALENVS_IN_PROJECT=true
source "$(poetry env info --path)/bin/activate"

python3 train/train.py | tee data/model/console.log
