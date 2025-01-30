doctl compute droplet create \
    --image ubuntu-24-10-x64 \
    --size s-2vcpu-4gb \
    --region fra1 \
    --user-data-file do-startup.sh \
    --ssh-keys "44677180" \
    --vpc-uuid f70e3a40-dc84-11e8-8b13-3cfdfea9f160 \
    ubuntu-s-2vcpu-4gb-fra1-01

count=0
timeout=90

while ! doctl compute d list | awk 'NR > 1 {print $3}'; do
   if [ $count -ge $timeout ]; then
       echo -e "\rHost timeout ($timeout sec). Exit."
       exit 1
   fi
   echo -ne "\rWaiting for host: $((++count)) sec for $timeout"
   sleep 1
done

echo -e "\rHost created!"


# doctl compute d list | awk 'NR > 1 {print $3}'