# Open Agent Hackathon 2026 — one-click runnable demo image.
# Pure stdlib: the image needs no pip install and no network to run the demo.
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml requirements.txt README.md ./
COPY src/ ./src/
COPY tests/ ./tests/
COPY examples/ ./examples/
COPY docs/ ./docs/

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    SQUAD_WORKSPACE=/tmp/agent-squad-workspace

# Deterministic, offline, zero-credential demo (this is what judges run).
CMD ["python", "-m", "squad.cli", "demo"]
