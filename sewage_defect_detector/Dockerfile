FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

CMD ["python", "onnx_inference.py", \
     "--model",     "/models/model_int8.onnx", \
     "--image_dir", "/data/images", \
     "--threshold", "0.4", \
     "--output",    "/output/predictions.csv"]