#!/bin/sh
poetry env list
python3 train/train.py | tee data/model/console.log
