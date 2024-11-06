# Python 3.12 slim 이미지 사용
FROM python:3.12.5-slim

RUN pip install --upgrade pip setuptools

# 빌드 도구와 dlib 설치를 위한 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    g++ \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get clean && apt-get -y update && apt-get install -y build-essential cmake libopenblas-dev liblapack-dev libopenblas-dev liblapack-dev

# dlib 및 다른 Python 패키지 설치
RUN pip install --no-cache-dir dlib

COPY . .

# 나머지 Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
