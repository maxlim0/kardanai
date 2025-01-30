ARG TARGETPLATFORM=linux/amd64
FROM python:3.10.16-slim
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app
COPY . .
VOLUME /app/data/model
RUN sh deploy/do-startup-dockerfile.sh && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /root/.cache/
RUN chmod +x deploy/entrypoint.sh
ENTRYPOINT ["deploy/entrypoint.sh"]

# apt-get install nvidia-container-toolkit
# это надо для работы нвидия в докере, надеюсь эта херня уже есть на хосте 
# иначе надо после установки делать  systemctl restart docker

# DOCKER_BUILDKIT=1 docker build -t maxsolyaris/kardanai:latest .
# DOCKER_BUILDKIT=1 docker build --platform linux/amd64 .
# docker build -t maxsolyaris/kardanai:latest .


# multi plarform
# docker buildx build --platform linux/amd64,linux/arm64 \
#   -t maxsolyaris/kardanai:latest \
#   --push .