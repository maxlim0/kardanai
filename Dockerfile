ARG TARGETPLATFORM=linux/amd64
FROM nvidia/cuda:11.8.0-base-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app
COPY . .
VOLUME /app/data/model
RUN sh deploy/do-startup.sh && chmod +x deploy/entrypoint.sh
ENTRYPOINT ["deploy/entrypoint.sh"]

# apt-get install nvidia-container-toolkit
# это надо для работы нвидия в докере, надеюсь эта херня уже есть на хосте 
# иначе надо после установки делать  systemctl restart docker

# DOCKER_BUILDKIT=1 docker build -t maxsolyaris/kardanai:latest .
# DOCKER_BUILDKIT=1 docker build --platform linux/amd64 .
# docker build -t maxsolyaris/kardanai:latest .