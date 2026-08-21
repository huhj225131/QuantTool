# Base Image với PyTorch & CUDA 12.1 sẵn có để hỗ trợ tối đa GPU/NVIDIA ModelOpt
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Thiết lập Environment Variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT=7860

# Cài đặt các thư viện hệ thống cần thiết (OpenCV GL libraries, git)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Thư mục làm việc
WORKDIR /app

# Copy requirements.txt để cache Docker layer
COPY requirements.txt /app/

# Cài đặt Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code và cấu hình
COPY setup.py /app/
COPY app.py /app/
COPY configs/ /app/configs/
COPY src/ /app/src/
COPY ui/ /app/ui/

# Tạo thư mục outputs và logs
RUN mkdir -p /app/outputs /app/logs

# Port mặc định của Gradio UI
EXPOSE 7860

# Lệnh mặc định khi chạy container
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]
