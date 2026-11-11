FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.4.1 torchvision==0.19.1

COPY requirements-api.txt pyproject.toml ./
RUN python -m pip install --no-cache-dir -r requirements-api.txt

COPY app ./app
COPY src ./src
RUN python -m pip install --no-cache-dir --no-deps .

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
