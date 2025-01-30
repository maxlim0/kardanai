# install doctl
# doctl compute d list | awk 'NR > 1 {print $3}'
doctl compute d list | awk 'NR > 1 {print $2}' | xargs -I {} doctl compute d delete -f {}