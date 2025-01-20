#!/bin/bash

# Обновление системы
apt-get update
apt-get upgrade -y

# Установка необходимых зависимостей
apt-get install -y python3 python3-pip software-properties-common

# Установка Ansible
apt-add-repository --yes --update ppa:ansible/ansible
apt-get install -y ansible git

# Создание директории для Ansible
mkdir -p /etc/ansible

# Создание inventory файла
echo "localhost ansible_connection=local" > /etc/ansible/hosts

git clone https://github.com/maxlim0/kardanai.git
cd kardanai.git

ansible-playbook absible.yml