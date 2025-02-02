# Эта логика нужна дла GihHub CI что бы понять когда закончился провиженинг 
# дроплета и окружение готово к запуску основного приложения в докер контейнере
#
#

#!/bin/bash

SERVER_IP="your_droplet_ip"
SSH_USER="root"  # или ваш пользователь

check_startup() {
    status=$(ssh -o StrictHostKeyChecking=no $SSH_USER@$DROPLET_IP 'cloud-init status')
    if [[ $status == *"status: done"* ]]; then
        return 0  # скрипт завершен
    else
        return 1  # скрипт все еще выполняется
    fi
}


timeout=600
start_time=$(date +%s)

while true; do
    if check_startup; then
        echo "Startup script завершен"
        break
    fi
    
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    
    if [ $elapsed -gt $timeout ]; then
        echo "Превышено время ожидания (${timeout}s)"
        exit 1
    fi
    
    echo "Ожидание завершения startup script... (прошло ${elapsed}s)"
    sleep 20 
done