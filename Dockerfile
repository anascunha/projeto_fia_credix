FROM python:3.13-slim

WORKDIR /app

COPY requirements-pipeline.txt /app/requirements-pipeline.txt
COPY app/requirements.txt /app/app/requirements.txt

RUN pip install --no-cache-dir -r /app/app/requirements.txt

COPY credix/ /app/credix/

COPY app/ /app/app/

COPY models/ /app/models/

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
