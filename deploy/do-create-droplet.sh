if [ "$1" == "GPU" ]; then
    echo "Creating GPU instance..."
    doctl compute droplet create \
        --region nyc2 \
        --image gpu-h100x1-base \
        --size gpu-h100x1-80gb \
        --user-data-file do-startup.sh \
        --ssh-keys "44677180" \
        ubuntu-gpu-h100x1-nyc1
else
    echo "Creating standard instance..."
    doctl compute droplet create \
        --image ubuntu-24-04-x64 \
        --size s-2vcpu-4gb \
        --region fra1 \
        --user-data-file do-startup.sh \
        --ssh-keys "44677180" \
        --vpc-uuid f70e3a40-dc84-11e8-8b13-3cfdfea9f160 \
        ubuntu-s-2vcpu-4gb-fra1-01
fi

count=0
timeout=90

while true; do
    ip=$(doctl compute d list | awk 'NR > 1 {print $3}')
    # Проверяем, что строка похожа на IP-адрес с помощью регулярного выражения
    if [[ $ip =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo -e "\rHost created! IP: $ip"
        sleep 40 # ждем поднятия сервисов
        break
    fi
    
    if [ $count -ge $timeout ]; then
        echo -e "\rHost timeout ($timeout sec). Exit."
        exit 1
    fi
    
    echo -ne "\rWaiting for host: $((++count)) sec for $timeout"
    sleep 1
done

PROJECT_DIR="/app"

# copy docker hub pwd
if [[ "$(hostname)" == "hole.local" ]]; then
    echo "Detected as hole.local system"
    ssh -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa_MacBookHoleGithub root@$(doctl compute d list | awk 'NR > 1 {print $3}') "mkdir -p /tmp/app/data/config"
    scp -r -i ~/.ssh/id_rsa_MacBookHoleGithub -o StrictHostKeyChecking=no \
        /Users/max/PycharmProjects/Topic/data/config root@$(doctl compute d list | awk 'NR > 1 {print $3}'):/tmp/$PROJECT_DIR/data/
elif env | grep -q "^GITHUB"; then
        echo "Detected as GITHUB system"
        echo "dockerhub_password: {{ secrets.DOCKERHUB_PASSWORD }}" > vars.yml
        echo "dockerhub_username: {{ secrets.DOCKERHUB_USERNAME }}" >> vars.yml
        # # copy docker hub pwd
        ssh -o StrictHostKeyChecking=no root@$DROPLET_IP "mkdir -p /tmp/app/data/config"
        scp -r -o StrictHostKeyChecking=no \
            vars.yml root@$DROPLET_IP:/tmp/$PROJECT_DIR/data/config
        echo $(doctl compute d list | awk 'NR > 1 {print $3}') > droplet_ip.txt
else
    echo "ERROR: System not detected."
fi

# doctl compute d list | awk 'NR > 1 {print $3}'