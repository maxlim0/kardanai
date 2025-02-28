#!/bin/bash

# depricated 
# Отказался от этой логики, тк startup безусловно должен завершится в success 
# иначе провиженинг сломан и его надо чинить

# Эта логика нужна дла GihHub CI что бы понять когда закончился провиженинг 
# дроплета и окружение готово к запуску основного приложения в докер контейнере

CONFIG_FILE="/tmp/app/data/config/ansible/vars.yml"
FLOW_CONTROL_GITHUB_TOKEN=$(grep "flow_control_github_token:" $CONFIG_FILE | cut -d' ' -f2)

check_startup() {
    status=$(cloud-init status)
    if [[ $status == *"status: done"* ]]; then
        return 0
    elif [[ $status == *"status: running"* ]]; then
        return 1  
    elif [[ $status == *"status: error"* ]]; then
        echo "cloud-init завершился с ошибкой"
        
        # Send status to GitHub Actions
        curl -X POST https://api.github.com/repos/maxlim0/kardanai/dispatches \
            -H 'Accept: application/vnd.github.everest-preview+json' \
            -H "Authorization: Bearer ${FLOW_CONTROL_GITHUB_TOKEN}" \
            --data '{"event_type": "cloud-init-status", "client_payload": { "status": "error"}}'        
        exit 1
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
        
        # Send status to GitHub Actions 
        curl -X POST https://api.github.com/repos/maxlim0/kardanai/dispatches \
            -H 'Accept: application/vnd.github.everest-preview+json' \
            -u "${FLOW_CONTROL_GITHUB_TOKEN}" \
            --data '{"event_type": "cloud-init-status", "client_payload": { "status": "ready"}}'
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