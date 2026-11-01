FROM python:3.12-slim

WORKDIR /app

# CPU-only PyTorch keeps the image much smaller than the default CUDA build
RUN pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["./run_pipeline.sh"]
