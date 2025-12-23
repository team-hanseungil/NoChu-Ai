FROM ubuntu:22.04

WORKDIR /app

# 기본 패키지 + Python 설치
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    gcc \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# python alias만 설정 (pip은 건드리지 않음)
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# 의존성 정의 파일 복사
COPY pyproject.toml .

# pip 업그레이드 및 의존성 설치
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir .

# 애플리케이션 소스 코드 복사
COPY . .

EXPOSE 8080

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
