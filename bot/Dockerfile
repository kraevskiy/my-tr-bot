FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=UTC

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY smcbot ./smcbot
COPY tests ./tests
COPY news ./news
COPY experiments*.json pyproject.toml README.md STRATEGY.md ARCHITECTURE.md ./

RUN useradd --create-home --uid 1000 bot && mkdir -p /app/data /app/results \
    && chown -R bot:bot /app
USER bot

VOLUME ["/app/data", "/app/results"]
# one entry point; the command (backtest, experiments, analyze, news, sessions, presets, config) is the first argument
ENTRYPOINT ["python", "-m", "smcbot"]
CMD ["backtest"]
