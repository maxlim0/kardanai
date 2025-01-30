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
echo "localhost ansible_connection=local" >c

# Создание ansible.cfg с настройками логирования
cat << EOF > /etc/ansible/ansible.cfg
[defaults]
log_path = /var/log/ansible.log
EOF

# Создание директории для логов и установка прав
touch /var/log/ansible.log
chmod 666 /var/log/ansible.log

if [ "$(hostname)" != "hole.local"]; then
    git clone https://github.com/maxlim0/kardanai.git $PROJECT_DIR
    ansible-playbook deploy/ansible-dockerfile.yml 

else
    ansible-playbook deploy/ansible-host-startup.yml
fi
