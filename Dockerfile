FROM --platform=linux/amd64 docker.1ms.run/pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

# 排除 torch 避免重复安装
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    grep -v "^torch==" requirements.txt > requirements-docker.txt && \
    pip install --no-cache-dir -r requirements-docker.txt && \
    rm requirements-docker.txt

WORKDIR /App

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PROJ_PATH=/App
ENV PYTHONPATH=/App/src
ENV DATA_DIR_PATH=Data

CMD ["bash"]
