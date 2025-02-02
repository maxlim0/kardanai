#!/bin/bash

# Эта логика нужна дла GihHub CI что бы понять когда закончился провиженинг 
# дроплета и окружение готово к запуску основного приложения в докер контейнере

#!/bin/bash

SSH_USER="root"

if [ -z "$1" ]; then
    echo "Ошибка: IP-адрес не передан в функцию"
    exit 1
fi

IP_ADDRESS="$1"

check_startup() {
    local ip="$1"
    echo "check_startup ip: ${ip}"
    status=$(ssh -o StrictHostKeyChecking=no $SSH_USER@${ip} 'cloud-init status')
    if [[ $status == *"status: done"* ]]; then
        return 0
    elif [[ $status == *"status: running"* ]]; then
        return 1
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
    if check_startup "$IP_ADDRESS"; then
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