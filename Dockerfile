# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY super_dooper_ibkr_bot.py ./

ENV IB_HOST=127.0.0.1 \
    IB_PORT=7497 \
    IB_CLIENT_ID=42 \
    ENABLE_XRP=true \
    MAX_WSB_POSTS=50

CMD ["python", "super_dooper_ibkr_bot.py"]
