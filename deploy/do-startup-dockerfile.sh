#!/bin/bash

PROJECT_DIR="/app"

# Установка необходимых зависимостей
apt-get update && \
apt-get install -y ansible git

# Создание директории для Ansible
mkdir -p /etc/ansible

# Создание inventory файла
echo "localhost ansible_connection=local" > /etc/ansible/hosts

# Создание ansible.cfg с настройками логирования
cat << EOF > /etc/ansible/ansible.cfg
[defaults]
log_path = /var/log/ansible.log
EOF

# Создание директории для логов и установка прав
touch /var/log/ansible.log
chmod 666 /var/log/ansible.log

ansible-playbook "$PROJECT_DIR/deploy/ansible-dockerfile.yml"

# TODO
# два одинаковых скрипта do-startup.sh и do-startup-dockerfile.sh
# разница в подготовке окружения, если это хост машина надо установить докер
# если это уже докер надо установить поетри и проект зависимости
# разница только в ансибл плейбуке
# надо либо сделать один скрипт с логикой определения контейнер это или хост
# или засунуть текущую логику в докер файл, но тогда тоже будет дубль
#  [ -f /.dockerenv ] почему-то эта проверка не работает на проимежуточных контейнерах и невозможно определить 