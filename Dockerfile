FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo-dev zlib1g-dev \
  && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
COPY src /app/src
RUN pip install -U pip && pip install -e .

# CD job will create a local ./model folder (best.pt + labels.json) before docker build
COPY model /model

ENV MODEL_DIR=/model
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "mmbiometric.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
