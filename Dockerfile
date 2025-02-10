ARG TARGETPLATFORM=linux/amd64
FROM maxsolyaris/kardanai:base
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:$PATH"
ENV POETRY_VIRTUALENVS_PATH=/root/.venv
ENV PYTHONUNBUFFERED=1
EXPOSE 22
WORKDIR /app
COPY . .
VOLUME /app/data/model
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    mkdir -p /root/.ssh && \
    mkdir -p /run/sshd && \
    chmod 700 /root/.ssh && \
    echo "${PUBLIC_KEY}" >> /root/.ssh/authorized_keys && \
    chmod 600 /root/.ssh/authorized_keys

ENTRYPOINT ["/bin/bash", "deploy/entrypoint.sh"]


# docker buildx build --platform linux/amd64 -t maxsolyaris/kardanai:latest --push .