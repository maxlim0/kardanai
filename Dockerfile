FROM nvidia/cuda:11.8.0-base-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app
COPY . .
VOLUME /app/data/model
RUN sh deploy/ansible-dockerfile.sh
ENTRYPOINT ["deploy/entrypoint.sh"]

# apt-get install nvidia-container-toolkit
# это надо для работы нвидия в докере, надеюсь эта херня уже есть на хосте 
# иначе надо после установки делать  systemctl restart docker
