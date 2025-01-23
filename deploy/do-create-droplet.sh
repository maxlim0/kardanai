doctl compute droplet create \
    --image ubuntu-24-10-x64 \
    --size s-2vcpu-4gb \
    --region fra1 \
    --user-data-file do-startup.sh \
    --ssh-keys "44677180" \
    --vpc-uuid f70e3a40-dc84-11e8-8b13-3cfdfea9f160 \
    ubuntu-s-2vcpu-4gb-fra1-01