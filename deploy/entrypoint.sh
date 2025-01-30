#!/bin/bash
poetry env list
poetry shell
python3 train/train.py | tee data/model/console.log
