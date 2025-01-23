#!/bin/bash

PROJECT_DIR="/root/kardanai"

# Установка необходимых зависимостей
apt-get install -y ansible-core git

# Создание директории для Ansible
mkdir -p /etc/ansible

# Создание inventory файла
echo "localhost ansible_connection=local" > /etc/ansible/hosts

git clone https://github.com/maxlim0/kardanai.git $PROJECT_DIR
cd $PROJECT_DIR

ansible-playbook ansible.yml