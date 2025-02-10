#!/bin/sh

service ssh start

sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    mkdir -p /root/.ssh && \
    mkdir -p /run/sshd && \
    chmod 700 /root/.ssh && \
    echo "${PUBLIC_KEY}" >> /root/.ssh/authorized_keys && \
    chmod 600 /root/.ssh/authorized_keys

poetry env list

export POETRY_VIRTUALENVS_IN_PROJECT=true
source "$(poetry env info --path)/bin/activate"

python3 -u train/train_v2.py 2>&1 | tee data/model/console.log
