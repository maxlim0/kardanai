# Эта логика нужна дла GihHub CI что бы понять когда закончился провиженинг 
# дроплета и окружение готово к запуску основного приложения в докер контейнере
#
#

#!/bin/bash

SSH_USER="root"  # или ваш пользователь

check_startup() {
    status=$(ssh -o StrictHostKeyChecking=no $SSH_USER@$1 'cloud-init status')
    echo "check_startup ip: ${$1}"
    if [[ $status == *"status: done"* ]]; then
        return 0  # скрипт завершен
    elif [[ $status == *"status: running"* ]]; then
        return 1  # скрипт все еще выполняется
    elif [[ $status == *"status: error"* ]]; then
        echo "cloud-init завершился с ошибкой"
        return 2
    else
        echo $status
        echo "cloud-init status UNKNOWN waiting..."
        return 3
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