FROM python:3.11-slim

# Repo root becomes WORKDIR
WORKDIR /Sewer-Defects-Detector

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

# Copy the entire working package
COPY sewage_defect_detector/ sewage_defect_detector/

WORKDIR /Sewer-Defects-Detector/sewage_defect_detector

ENV PYTHONUNBUFFERED=1
# So that `from src.xxx` imports resolve correctly
ENV PYTHONPATH=/Sewer-Defects-Detector/sewage_defect_detector

CMD ["python", "onnx_inference.py", \
     "--model",     "/models/model_int8.onnx", \
     "--image_dir", "/data/images", \
     "--threshold", "0.4", \
     "--output",    "/output/predictions.csv"]