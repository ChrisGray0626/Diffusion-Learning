FROM --platform=linux/amd64 docker.1ms.run/pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

# 排除 torch 避免重复安装
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    grep -v "^torch==" requirements.txt > requirements-docker.txt && \
    pip install --no-cache-dir -r requirements-docker.txt && \
    rm requirements-docker.txt

#RUN apt-get update && apt-get install -y \
#    libgdal-dev \
#    && rm -rf /var/lib/apt/lists/*

WORKDIR /App

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PROJ_PATH=/App \
    DATA_DIR_PATH=/Data

ENV PYTHONPATH=/App

CMD ["bash"]
