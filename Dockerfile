FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Python 3.10 설치 (22.04 기본)
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3-pip \
    git curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python

# uv 설치
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# uv 기본 환경 변수
ENV UV_PROJECT_ENVIRONMENT="/uv-venv" \
    UV_LINK_MODE=copy

WORKDIR /app

# 아직 코드 없으므로 최소 pyproject만
COPY pyproject.toml ./

# 환경만 살아있는지 확인 (실패해도 OK)
RUN uv sync || true

# venv PATH
ENV PATH="/uv-venv/bin:$PATH"

CMD ["bash"]
