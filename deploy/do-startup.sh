#!/bin/bash

PROJECT_DIR="/app"

# Установка необходимых зависимостей
apt-get update && \
apt-get install -y software-properties-common && \
add-apt-repository --yes --update ppa:ansible/ansible && \
apt-get install -y ansible git

# Создание директории для Ansible
mkdir -p /etc/ansible

# Создание inventory файла
echo "localhost ansible_connection=local" > /etc/ansible/hosts

git clone https://github.com/maxlim0/kardanai.git $PROJECT_DIR
cd $PROJECT_DIR/deploy

ansible-playbook ansible-host-startup.yml


# # copy dataset and config.py
# scp -r -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no \
#     data/export root@$(doctl compute d list | awk 'NR > 1 {print $3}'):$PROJECT_DIR/deploy/ \
#     config.py root@$(doctl compute d list | awk 'NR > 1 {print $3}'):$PROJECT_DIR/
